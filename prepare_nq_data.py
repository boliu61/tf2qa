# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Converts an NQ dataset file to tf examples.

Notes:
  - this program needs to be run for every NQ training shard.
  - this program only outputs the first n top level contexts for every example,
    where n is set through --max_contexts.
  - the restriction from --max_contexts is such that the annotated context might
    not be present in the output examples. --max_contexts=8 leads to about
    85% of examples containing the correct context. --max_contexts=48 leads to
    about 97% of examples containing the correct context.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import random
import json
import tensorflow as tf
import re

import collections
import copy
import enum

from nltk.tokenize import sent_tokenize
from bert import tokenization

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_float(
    "sos_proportion", -1,
    "do sos to how much eligible samples")

flags.DEFINE_float(
    "include_unknowns", -1.0,
    "If positive, probability of including answers of type `UNKNOWN`.")

flags.DEFINE_float(
    "include_unknowns_answerable", 0.02,
    "If positive, probability of including answers of type `UNKNOWN`.")

flags.DEFINE_float(
    "include_unknowns_unanswerable", 0.02,
    "If positive, probability of including answers of type `UNKNOWN`.")

flags.DEFINE_integer(
    "skip_n_records", 0,
    "skip this many records, for debug")

flags.DEFINE_bool(
    "short_ans_only", False,
    "Whether to only use short answers as target in train")

flags.DEFINE_integer(
    "shard", 0,
    "0 to 49")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_string(
    "tfrecord_dir", None,
    "Directory to save generated tfrecords")


flags.DEFINE_string(
    "input_jsonl", None,
    "Gzipped files containing NQ examples in Json format, one per line.")

flags.DEFINE_string("output_tfrecord", None,
                    "Output tf record file with all features extracted.")

flags.DEFINE_bool(
    "is_training", True,
    "Whether to prepare features for training or for evaluation. Eval features "
    "don't include gold labels, but include wordpiece to html token maps.")


flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")


flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal NQ evaluation.")

flags.DEFINE_boolean(
    "skip_nested_contexts", True,
    "Completely ignore context that are not top level nodes in the page.")

flags.DEFINE_integer("task_id", 0,
                     "Train and dev shard to read from and write to.")

flags.DEFINE_integer("max_contexts", 48,
                     "Maximum number of contexts to output for an example.")

flags.DEFINE_integer(
    "max_position", 50,
    "Maximum context position for which to generate special tokens.")

flags.DEFINE_integer(
    "max_examples", 0,
    "If positive, stop once these many examples have been converted.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")


TextSpan = collections.namedtuple("TextSpan", "token_positions text")


class AnswerType(enum.IntEnum):
  """Type of NQ answer."""
  UNKNOWN = 0
  YES = 1
  NO = 2
  SHORT = 3
  LONG = 4


class Answer(collections.namedtuple("Answer", ["type", "text", "offset"])):
  """Answer record.

  An Answer contains the type of the answer and possibly the text (for
  long) as well as the offset (for extractive).
  """

  def __new__(cls, type_, text=None, offset=None):
    return super(Answer, cls).__new__(cls, type_, text, offset)


class NqExample(object):
  """A single training/test example."""

  def __init__(self,
               example_id,
               qas_id,
               questions,
               doc_tokens,
               doc_tokens_map=None,
               answer=None,
               start_position=None,
               end_position=None):
    self.example_id = example_id
    self.qas_id = qas_id
    self.questions = questions
    self.doc_tokens = doc_tokens
    self.doc_tokens_map = doc_tokens_map
    self.answer = answer
    self.start_position = start_position
    self.end_position = end_position


def get_candidate_type(e, idx):
  """Returns the candidate's type: Table, Paragraph, List or Other."""
  c = e["long_answer_candidates"][idx]
  first_token = e["document_tokens"][c["start_token"]]["token"]
  if first_token == "<Table>":
    return "Table"
  elif first_token == "<P>":
    return "Paragraph"
  elif first_token in ("<Ul>", "<Dl>", "<Ol>"):
    return "List"
  elif first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
    return "Other"
  else:
    tf.logging.warning("Unknoww candidate type found: %s", first_token)
    return "Other"

def add_candidate_types_and_positions(e):
  """Adds type and position info to each candidate in the document."""
  counts = collections.defaultdict(int)
  for idx, c in candidates_iter(e):
    context_type = get_candidate_type(e, idx)  # Paragraph, Table, List, Other
    if counts[context_type] < FLAGS.max_position:
      counts[context_type] += 1
    c["type_and_position"] = "[%s=%d]" % (context_type, counts[context_type])


def get_candidate_type_and_position(e, idx):
  """Returns type and position info for the candidate at the given index."""
  if idx == -1:
    return "[NoLongAnswer]"
  else:
    return e["long_answer_candidates"][idx]["type_and_position"]


def gold_has_long_answer(gold_label_list):
  """Gets vote from multi-annotators for judging if there is a long answer."""

  gold_has_answer = gold_label_list and (sum([
    not label.long_answer_span.is_null_span()  # long answer not null
    for label in gold_label_list  # for each annotator
  ]) >= 2)

  return gold_has_answer


def has_long_answer(a):
  return (a["long_answer"]["start_token"] >= 0 and
          a["long_answer"]["end_token"] >= 0)


def get_first_annotation(e):
  """Returns the first short or long answer in the example.

  Args:
    e: (dict) annotated example.

  Returns:
    annotation: (dict) selected annotation
    annotated_idx: (int) index of the first annotated candidate.
    annotated_sa: (tuple) char offset of the start and end token
        of the short answer. The end token is exclusive.
  """
  positive_annotations = sorted(
      [a for a in e["annotations"] if has_long_answer(a)],
      key=lambda a: a["long_answer"]["candidate_index"])

  for a in positive_annotations:
    if a["short_answers"]:
      idx = a["long_answer"]["candidate_index"]
      start_token = a["short_answers"][0]["start_token"]
      end_token = a["short_answers"][-1]["end_token"]
      return a, idx, (token_to_char_offset(e, idx, start_token),
                      token_to_char_offset(e, idx, end_token) - 1)

  for a in positive_annotations:
    idx = a["long_answer"]["candidate_index"]
    return a, idx, (-1, -1)

  return None, -1, (-1, -1)

def token_to_char_offset(e, candidate_idx, token_idx):
  """Converts a token index to the char offset within the candidate."""
  c = e["long_answer_candidates"][candidate_idx]
  char_offset = 0
  for i in range(c["start_token"], token_idx):
    t = e["document_tokens"][i]
    if not t["html_token"]:
      token = t["token"].replace(" ", "")
      char_offset += len(token) + 1
  return char_offset

def get_text_span(example, span):
  """Returns the text in the example's document in the given token span."""
  token_positions = []
  tokens = []
  for i in range(span["start_token"], span["end_token"]):
    t = example["document_tokens"][i]
    if not t["html_token"]:
      token_positions.append(i)
      token = t["token"].replace(" ", "")
      tokens.append(token)
  return TextSpan(token_positions, " ".join(tokens))


def get_candidate_text(e, idx):
  """Returns a text representation of the candidate at the given index."""
  # No candidate at this index.
  if idx < 0 or idx >= len(e["long_answer_candidates"]):
    return TextSpan([], "")

  # This returns an actual candidate.
  return get_text_span(e, e["long_answer_candidates"][idx])


def candidates_iter(e):
  """Yield's the candidates that should not be skipped in an example."""
  for idx, c in enumerate(e["long_answer_candidates"]):
    if should_skip_context(e, idx):
      continue
    yield idx, c

def should_skip_context(e, idx):
  if (FLAGS.skip_nested_contexts and
      not e["long_answer_candidates"][idx]["top_level"]):
    return True
  elif not get_candidate_text(e, idx).text.strip():
    # Skip empty contexts.
    return True
  else:
    return False

def create_example_from_jsonl(line):
  """Creates an NQ example from a given line of JSON."""
  e = json.loads(line, object_pairs_hook=collections.OrderedDict)
  # assert len(e['annotations'])==1
  add_candidate_types_and_positions(e)
  annotation, annotated_idx, annotated_sa = get_first_annotation(e)

  # annotated_idx: index of the first annotated context, -1 if null.
  # annotated_sa: short answer start and end char offsets, (-1, -1) if null.
  question = {"input_text": e["question_text"]}
  answer = {
      "candidate_id": annotated_idx,
      "span_text": "",
      "span_start": -1,
      "span_end": -1,
      "input_text": "long",
  }

  if annotated_idx != -1:
    # 短 gt 不是top level
    gt_candidate = e['long_answer_candidates'][annotated_idx]
    if not gt_candidate['top_level']:
      i = annotated_idx
      while(not e['long_answer_candidates'][i]['top_level']):
        i = i-1
      containing_cand = e['long_answer_candidates'][i]
      char_diff = sum([len(e['document_tokens'][j]['token'])+1 for j in range(containing_cand['start_token'],gt_candidate['start_token']) if not e['document_tokens'][j]['html_token']])
      if annotated_sa!=(-1,-1): annotated_sa = [x+char_diff for x in annotated_sa]
      annotated_idx = i
      annotation['long_answer']['candidate_index'] = i
      for field in ['start_byte','end_byte','start_token','end_token']:
        annotation['long_answer'][field] = containing_cand[field]
      # print('\n fixed a non-top-level gt')

  # Yes/no answers are added in the input text.
  if annotation is not None:
    assert annotation["yes_no_answer"] in ("YES", "NO", "NONE")
    if annotation["yes_no_answer"] in ("YES", "NO"):
      answer["input_text"] = annotation["yes_no_answer"].lower()

  # Add a short answer if one was found
  if annotated_sa != (-1, -1):
    answer["input_text"] = "short"
    span_text = get_candidate_text(e, annotated_idx).text
    answer["span_text"] = span_text[annotated_sa[0]:annotated_sa[1]]
    answer["span_start"] = annotated_sa[0]
    answer["span_end"] = annotated_sa[1]
    expected_answer_text = get_text_span(
        e, {
            "start_token": annotation["short_answers"][0]["start_token"],
            "end_token": annotation["short_answers"][-1]["end_token"],
        }).text
    try:
      assert expected_answer_text == answer["span_text"], (expected_answer_text,
                                                         answer["span_text"])
    except:
      print('\ngt still wrong after fixed top level')
      return None

  # Add a long answer if one was found
  elif annotation and annotation["long_answer"]["candidate_index"] >= 0:
    answer["span_text"] = get_candidate_text(e, annotated_idx).text
    answer["span_start"] = 0
    answer["span_end"] = len(answer["span_text"])

  context_idxs = [-1]
  context_list = [{"id": -1, "type": get_candidate_type_and_position(e, -1)}]  # [NoLongAnswer]
  context_list[-1]["text_map"], context_list[-1]["text"] = (
      get_candidate_text(e, -1))
  for idx, _ in candidates_iter(e):
    context = {"id": idx, "type": get_candidate_type_and_position(e, idx)}
    context["text_map"], context["text"] = get_candidate_text(e, idx)
    context_idxs.append(idx)
    context_list.append(context)
    if len(context_list) >= FLAGS.max_contexts:
      # print("context_list = {:d}".format(len(context_list)))
      break

  # Assemble example.
  example = {
      "name": e["document_title"],
      "id": str(e["example_id"]),
      "questions": [question],
      "answers": [answer],
      "has_correct_context": annotated_idx in context_idxs
  }

  single_map = []
  single_context = []
  offset = 0
  found_context = False
  for context in context_list:
    single_map.extend([-1, -1])
    single_context.append("[ContextId=%d] %s" %
                          (context["id"], context["type"]))
    offset += len(single_context[-1]) + 1
    if context["id"] == annotated_idx:
      found_context = True
      answer["span_start"] += offset
      answer["span_end"] += offset

    # Many contexts are empty once the HTML tags have been stripped, so we
    # want to skip those.
    if context["text"]:
      single_map.extend(context["text_map"])
      single_context.append(context["text"])
      offset += len(single_context[-1]) + 1

  # gt not in first 48 contexts (only has long answers，not top level）
  # if not found_context:
  #   print("context['id'], annotated_idx = {:d}, {:d} \n".format(context["id"], annotated_idx))
    # return None

  example["contexts"] = " ".join(single_context)
  example["contexts_map"] = single_map
  if annotated_idx in context_idxs:
    expected = example["contexts"][answer["span_start"]:answer["span_end"]]

    # This is a sanity check to ensure that the calculated start and end
    # indices match the reported span text. If this assert fails, it is likely
    # a bug in the data preparation code above.
    assert expected == answer["span_text"], (expected, answer["span_text"])

  return example


# A special token in NQ is made of non-space chars enclosed in square brackets.
_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)


def tokenize(tokenizer, text, apply_basic_tokenization=False):
  """Tokenizes text, optionally looking up special tokens separately.

  Args:
    tokenizer: a tokenizer from bert.tokenization.FullTokenizer
    text: text to tokenize
    apply_basic_tokenization: If True, apply the basic tokenization. If False,
      apply the full tokenization (basic + wordpiece).

  Returns:
    tokenized text.

  A special token is any text with no spaces enclosed in square brackets with no
  space, so we separate those out and look them up in the dictionary before
  doing actual tokenization.
  """
  tokenize_fn = tokenizer.tokenize
  if apply_basic_tokenization:
    tokenize_fn = tokenizer.basic_tokenizer.tokenize
  tokens = []
  for token in text.split(" "):
    if _SPECIAL_TOKENS_RE.match(token):
      if token in tokenizer.vocab:
        tokens.append(token)
      else:
        tokens.append(tokenizer.wordpiece_tokenizer.unk_token)
    else:
      tokens.extend(tokenize_fn(token))
  return tokens


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               token_to_orig_map,
               input_ids,
               input_mask,
               segment_ids,
               start_position=None,
               end_position=None,
               answer_text="",
               answer_type=AnswerType.SHORT):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.token_to_orig_map = token_to_orig_map
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.start_position = start_position
    self.end_position = end_position
    self.answer_text = answer_text
    self.answer_type = answer_type


def convert_single_example(example, tokenizer, is_training, to_print=False):
  """Converts a single NqExample into a list of InputFeatures."""
  tok_to_example_doc_tokens_index = []
  orig_to_tok_index = []
  all_doc_tokens = []
  features = []
  for (i, token) in enumerate(example.doc_tokens):
    orig_to_tok_index.append(len(all_doc_tokens))
    sub_tokens = tokenize(tokenizer, token)
    tok_to_example_doc_tokens_index.extend([i] * len(sub_tokens))
    all_doc_tokens.extend(sub_tokens)
  # `tok_to_orig_index` maps wordpiece indices to indices of whitespace
  # tokenized word tokens in the contexts. The word tokens might themselves
  # correspond to word tokens in a larger document, with the mapping given
  # by `doc_tokens_map`.
  if example.doc_tokens_map:
    tok_to_orig_index = [
        example.doc_tokens_map[index] for index in tok_to_example_doc_tokens_index
    ]

  # QUERY
  query_tokens = []
  query_tokens.append("[Q]")
  query_tokens.extend(tokenize(tokenizer, example.questions[-1]))
  if len(query_tokens) > FLAGS.max_query_length:
    query_tokens = query_tokens[-FLAGS.max_query_length:]

  # ANSWER
  tok_start_position = 0
  tok_end_position = 0
  if is_training:
    tok_start_position = orig_to_tok_index[example.start_position]
    if example.end_position < len(example.doc_tokens) - 1:
      tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
    else:
      tok_end_position = len(all_doc_tokens) - 1

  # The -3 accounts for [CLS], [SEP] and [SEP]
  max_tokens_for_doc = FLAGS.max_seq_length - len(query_tokens) - 3

  # We can have documents that are longer than the maximum sequence length.
  # To deal with this we do a sliding window approach, where we take chunks
  # of up to our max length with a stride of `doc_stride`.
  _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
      "DocSpan", ["start", "length"])
  doc_spans = []
  start_offset = 0
  while start_offset < len(all_doc_tokens):
    length = len(all_doc_tokens) - start_offset
    length = min(length, max_tokens_for_doc)
    doc_spans.append(_DocSpan(start=start_offset, length=length))
    if start_offset + length == len(all_doc_tokens):
      break
    start_offset += min(length, FLAGS.doc_stride)

  for (doc_span_index, doc_span) in enumerate(doc_spans):
    tokens = []
    token_to_orig_map = {}
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    tokens.extend(query_tokens)
    segment_ids.extend([0] * len(query_tokens))
    tokens.append("[SEP]")
    segment_ids.append(0)

    for i in range(doc_span.length):
      split_token_index = doc_span.start + i
      token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
      tokens.append(all_doc_tokens[split_token_index])
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
    assert len(tokens) == len(segment_ids)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (FLAGS.max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)

    assert len(input_ids) == FLAGS.max_seq_length
    assert len(input_mask) == FLAGS.max_seq_length
    assert len(segment_ids) == FLAGS.max_seq_length

    start_position = None
    end_position = None
    answer_type = None
    answer_text = ""
    if is_training:
      doc_start = doc_span.start
      doc_end = doc_span.start + doc_span.length - 1
      # For training, if our document chunk does not contain an annotation
      # we throw it out, since there is nothing to predict.
      contains_an_annotation = (
          tok_start_position >= doc_start and tok_end_position <= doc_end)

      if example.answer.type == AnswerType.UNKNOWN:
        if (FLAGS.include_unknowns_unanswerable < 0 or
            random.random() > FLAGS.include_unknowns_unanswerable):
          continue
        start_position = 0
        end_position = 0
        answer_type = AnswerType.UNKNOWN
      elif (not contains_an_annotation):
        # If an example has unknown answer type or does not contain the answer
        # span, then we only include it with probability --include_unknowns.
        # When we include an example with unknown answer type, we set the first
        # token of the passage to be the annotated short span.
        if (FLAGS.include_unknowns_answerable < 0 or
            random.random() > FLAGS.include_unknowns_answerable):
          continue
        start_position = 0
        end_position = 0
        answer_type = AnswerType.UNKNOWN
      else:
        doc_offset = len(query_tokens) + 2
        start_position = tok_start_position - doc_start + doc_offset
        end_position = tok_end_position - doc_start + doc_offset
        answer_type = example.answer.type

        # sentence order shuffle
        if answer_type == AnswerType.SHORT:
          ans_start_whitespace = tok_to_example_doc_tokens_index[tok_start_position]
          ans_end_whitespace = tok_to_example_doc_tokens_index[tok_end_position]
          doc_start_whitespace = tok_to_example_doc_tokens_index[doc_start]

          ## special case, first word got split
          offset_adj = doc_start - tok_to_example_doc_tokens_index.index(doc_start_whitespace)
          # if offset_adj>0:
          #   print(' ')
          doc_end_whitespace = tok_to_example_doc_tokens_index[doc_end]
          para = ' '.join( example.doc_tokens[doc_start_whitespace:(doc_end_whitespace+1)])
          ans_start_in_span = len(' '.join(example.doc_tokens[doc_start_whitespace:ans_start_whitespace]))
          ans_end_in_span = len(' '.join(example.doc_tokens[doc_start_whitespace:(ans_end_whitespace+1)]))

          ans = para[ans_start_in_span:ans_end_in_span].strip(' ')

          if para.count(ans) != 1:
            # print("para.count(ans) = {:d}".format(para.count(ans)))
            pass
          elif len(sent_tokenize(ans))>1:
            # print("short ans has more than 1 sentence")
            pass
          elif (FLAGS.sos_proportion < 0 or random.random() > FLAGS.sos_proportion):
            # print('skip')
            pass
          else:
            print('SOS')
            num_tok_before_ans = len(para[:ans_start_in_span].split(' '))

            prev_special_i, next_special_i = None, None

            for i, tok in enumerate(para.split(' ')):
              if _SPECIAL_TOKENS_RE.match(tok):
                # print(i, tok)
                if i < num_tok_before_ans:
                  if (prev_special_i is None) or (prev_special_i != i-1):
                    prev_special_i = i
                elif next_special_i is None:
                  next_special_i = i

            if prev_special_i is None: prev_special_i=0
            if next_special_i is None: next_special_i = len(para.split(' '))
            para_before = ' '.join(para.split(' ')[:prev_special_i])
            para_with_ans = ' '.join(para.split(' ')[prev_special_i:next_special_i])
            para_after = ' '.join(para.split(' ')[next_special_i:])

            first_two_special_tok = ''
            if len(para_before) > 0:
              if len(para_with_ans.split(' '))==1:
                print(f"ans_start_in_span = {ans_start_in_span}")
                print(f"ans = {ans}")
                print(f"num_tok_before_ans = {num_tok_before_ans}")
                print(f"prev_special_i = {prev_special_i}")
                print(f"next_special_i = {next_special_i}")
                print(f"para_before = {para_before}")
                print(f"para_with_ans = {para_with_ans}")
                print(f"para_after = {para_after}")

              if not (_SPECIAL_TOKENS_RE.match(para_with_ans.split(' ')[0]) and _SPECIAL_TOKENS_RE.match(para_with_ans.split(' ')[1])):
                print(' ')
              assert _SPECIAL_TOKENS_RE.match(para_with_ans.split(' ')[0]) and _SPECIAL_TOKENS_RE.match(para_with_ans.split(' ')[1])
              num_special_tok = 2
              if _SPECIAL_TOKENS_RE.match(para_with_ans.split(' ')[2]):
                num_special_tok += 1
              if _SPECIAL_TOKENS_RE.match(para_with_ans.split(' ')[3]):
                num_special_tok += 1
              first_two_special_tok = ' '.join(para_with_ans.split(' ')[:num_special_tok])
              para_with_ans = ' '.join(para_with_ans.split(' ')[num_special_tok:])

            sentences = sent_tokenize(para_with_ans)

            num_sent = len(sentences)
            if num_sent <= 2:
              # print('num sentence <= 2')
              pass
            else:
              # if paragraph containing short gt is at end of window, don't shuffle last sentence
              # if paragraph containing short gt is at beginning of window, don't shuffle first sentence
              if len(para_before)==0 and len(para_after)>0:
                random_idx = random.sample(range(1, num_sent), num_sent - 1)
                para_shuffled = ' '.join([sentences[i] for i in ([0] + random_idx)])
              elif len(para_before)==0 and len(para_after)==0:
                random_idx = random.sample(range(1, num_sent-1), num_sent - 2)
                para_shuffled = ' '.join([sentences[i] for i in ([0] + random_idx + [num_sent-1])])
              elif len(para_before)> 0 and len(para_after) == 0:
                random_idx = random.sample(range(0, num_sent-1), num_sent - 1)
                para_shuffled = ' '.join([sentences[i] for i in (random_idx + [num_sent - 1])])
              else:
                random_idx = random.sample(range(0, num_sent), num_sent)
                para_shuffled = ' '.join([sentences[i] for i in random_idx])

              para_sos = ' '.join([para_before, first_two_special_tok, para_shuffled, para_after]).strip(' ').replace('  ',' ')

              tokens_before_sos = tokenize(tokenizer, para)
              tokens_after_sos = tokenize(tokenizer, para_sos)
              assert sorted(tokens_before_sos) == sorted(tokens_after_sos)

              ## special case, first word got split by window
              # offset_adj = 0
              # if tokens[doc_offset].startswith('#'):
              # offset_adj = tokens_before_sos.index(tokens[doc_offset])
              tokens_before_sos = tokens_before_sos[offset_adj:]
              tokens_after_sos = tokens_after_sos[offset_adj:]

              tokens[doc_offset:(-1)] = tokens_after_sos[:(FLAGS.max_seq_length-1-doc_offset)]

              input_ids_before_sos = copy.deepcopy(input_ids)
              # input_ids[doc_offset:(doc_offset+len(tokens))] = tokenizer.convert_tokens_to_ids(tokens)
              input_ids = tokenizer.convert_tokens_to_ids(tokens) + padding
              if sorted(input_ids_before_sos) != sorted(input_ids):
                tmp1 = ' '.join([str(x) for x in input_ids_before_sos])
                tmp2 = ' '.join([str(x) for x in input_ids])
                print(' ')
              assert sorted(input_ids_before_sos) == sorted(input_ids)

              # if ans not in para_sos:
              #   print(' ')
              ans_pos_in_sos = para_sos.index(ans)
              end_minus_start = end_position - start_position
              ans_ids_before_sos = input_ids_before_sos[start_position:(end_position+1)]
              start_position = len(tokenize(tokenizer, para_sos[:ans_pos_in_sos])) - offset_adj + doc_offset
              end_position = start_position + end_minus_start
              ans_ids = input_ids[start_position:(end_position+1)]
              assert ans_ids == ans_ids_before_sos

        # if gt is long answer, sample 2% as null sample into training set, discard the rest
        if FLAGS.short_ans_only and answer_type != AnswerType.SHORT:
          if (FLAGS.include_unknowns < 0 or
                  random.random() > FLAGS.include_unknowns):
            continue
          start_position = 0
          end_position = 0
          answer_type = AnswerType.UNKNOWN

      answer_text = " ".join(tokens[start_position:(end_position + 1)])


    feature = InputFeatures(
        unique_id=-1,
        example_index=-1,
        doc_span_index=doc_span_index,
        token_to_orig_map=token_to_orig_map,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        start_position=start_position,
        end_position=end_position,
        answer_text=answer_text,
        answer_type=answer_type)

    features.append(feature)

  return features


def make_nq_answer(contexts, answer):
  """Makes an Answer object following NQ conventions.

  Args:
    contexts: string containing the context
    answer: dictionary with `span_start` and `input_text` fields

  Returns:
    an Answer object. If the Answer type is YES or NO or LONG, the text
    of the answer is the long answer. If the answer type is UNKNOWN, the text of
    the answer is empty.
  """
  start = answer["span_start"]
  end = answer["span_end"]
  input_text = answer["input_text"]

  if (answer["candidate_id"] == -1 or start >= len(contexts) or
      end > len(contexts)):
    answer_type = AnswerType.UNKNOWN
    start = 0
    end = 1
  elif input_text.lower() == "yes":
    answer_type = AnswerType.YES
  elif input_text.lower() == "no":
    answer_type = AnswerType.NO
  elif input_text.lower() == "long":
    answer_type = AnswerType.LONG
  else:
    answer_type = AnswerType.SHORT

  return Answer(answer_type, text=contexts[start:end], offset=start)


def read_nq_entry(entry, is_training):
  """Converts a NQ entry into a list of NqExamples."""

  def is_whitespace(c):
    return c in " \t\r\n" or ord(c) == 0x202F

  examples = []
  contexts_id = entry["id"]
  contexts = entry["contexts"]
  doc_tokens = []
  char_to_word_offset = []
  prev_is_whitespace = True
  for c in contexts:
    if is_whitespace(c):
      prev_is_whitespace = True
    else:
      if prev_is_whitespace:
        doc_tokens.append(c)
      else:
        doc_tokens[-1] += c
      prev_is_whitespace = False
    char_to_word_offset.append(len(doc_tokens) - 1)

  questions = []
  for i, question in enumerate(entry["questions"]):
    qas_id = "{}".format(contexts_id)
    question_text = question["input_text"]
    start_position = None
    end_position = None
    answer = None
    if is_training:
      answer_dict = entry["answers"][i]
      answer = make_nq_answer(contexts, answer_dict)
      if answer.type != AnswerType.UNKNOWN and  answer_dict['span_text'] != answer.text:
        # print('wrong gt')
        return []

      # For now, only handle extractive, yes, and no.
      if answer is None or answer.offset is None:
        continue
      start_position = char_to_word_offset[answer.offset]
      end_position = char_to_word_offset[answer.offset + len(answer.text) - 1]

      # Only add answers where the text can be exactly recovered from the
      # document. If this CAN'T happen it's likely due to weird Unicode
      # stuff so we will just skip the example.
      #
      # Note that this means for training mode, every example is NOT
      # guaranteed to be preserved.
      actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
      cleaned_answer_text = " ".join(
          tokenization.whitespace_tokenize(answer.text))
      if actual_text.find(cleaned_answer_text) == -1:
        tf.logging.warning("Could not find answer: '%s' vs. '%s'", actual_text,
                           cleaned_answer_text)
        continue

    questions.append(question_text)
    example = NqExample(
        example_id=int(contexts_id),
        qas_id=qas_id,
        questions=questions[:],
        doc_tokens=doc_tokens,
        doc_tokens_map=entry.get("contexts_map", None),
        answer=answer,
        start_position=start_position,
        end_position=end_position)
    examples.append(example)
  return examples


class CreateTFExampleFn(object):
  """Functor for creating NQ tf.Examples."""

  def __init__(self, is_training):
    self.is_training = is_training
    self.tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    # self.ans_type_cnt = collections.defaultdict(int)
    self.feat_cnt = []

  def process(self, example):
    """Coverts an NQ example in a list of serialized tf examples."""
    nq_examples = read_nq_entry(example, self.is_training)

    input_features = []
    for nq_example in nq_examples:
      input_features.extend(
          convert_single_example(nq_example, self.tokenizer, self.is_training, to_print=False))

    for input_feature in input_features:
      input_feature.example_index = int(example["id"])
      input_feature.unique_id = (
          input_feature.example_index + input_feature.doc_span_index)

      def create_int_feature(values):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(values)))

      features = collections.OrderedDict()
      features["unique_ids"] = create_int_feature([input_feature.unique_id])
      features["input_ids"] = create_int_feature(input_feature.input_ids)
      features["input_mask"] = create_int_feature(input_feature.input_mask)
      features["segment_ids"] = create_int_feature(input_feature.segment_ids)

      if self.is_training:
        features["start_positions"] = create_int_feature(
            [input_feature.start_position])
        features["end_positions"] = create_int_feature(
            [input_feature.end_position])
        features["answer_types"] = create_int_feature(
            [input_feature.answer_type])
      else:
        token_map = [-1] * len(input_feature.input_ids)
        for k, v in input_feature.token_to_orig_map.iteritems():
          token_map[k] = v
        features["token_map"] = create_int_feature(token_map)

      yield tf.train.Example(features=tf.train.Features(
          feature=features)).SerializeToString()




def get_examples(input_jsonl_pattern):
  for input_path in tf.gfile.Glob(input_jsonl_pattern):
    with gzip.open(input_path, 'r') as input_file:
      num_lines = sum([1 for _ in input_file])
    with gzip.open(input_path, 'r') as input_file:
      from tqdm import tqdm
      for line_idx,line in tqdm(enumerate(input_file),total=num_lines):
        if line_idx < FLAGS.skip_n_records: # for debug
          continue
        yield create_example_from_jsonl(line)


def main(_):
  shard = FLAGS.shard
  FLAGS.include_unknowns = 0.02

  if FLAGS.do_lower_case:
    FLAGS.vocab_file = "../bert-joint-baseline/vocab-nq.txt"
  else:
    FLAGS.vocab_file = "../vocab_cased-nq.txt"

  FLAGS.max_seq_length = 512
  FLAGS.input_jsonl = f'../data_nq/train/nq-train-{shard:02}.jsonl.gz'
  FLAGS.output_tfrecord = f'../data_nq/'+ FLAGS.tfrecord_dir + f'/nq-train.tfrecords-{shard:02}'

  examples_processed = 0
  num_examples_with_correct_context = 0
  creator_fn = CreateTFExampleFn(is_training=FLAGS.is_training)

  instances = []

  tf.logging.set_verbosity(tf.logging.INFO)

  from tqdm import tqdm
  for example in tqdm(get_examples(FLAGS.input_jsonl)):

    # gt not in first 48 contexts
    if example is None:
      continue

    for instance in creator_fn.process(example):
      instances.append(instance)
    if example["has_correct_context"]:
      num_examples_with_correct_context += 1
    # if examples_processed % 100 == 0:
    #   tf.logging.info("Examples processed: %d", examples_processed)
    examples_processed += 1
    if FLAGS.max_examples > 0 and examples_processed >= FLAGS.max_examples:
      break
  tf.logging.info("Examples with correct context retained: %d of %d",
                  num_examples_with_correct_context, examples_processed)

  random.shuffle(instances)
  with tf.python_io.TFRecordWriter(FLAGS.output_tfrecord) as writer:
    for instance in instances:
      writer.write(instance)


if __name__ == "__main__":
  tf.app.run()
