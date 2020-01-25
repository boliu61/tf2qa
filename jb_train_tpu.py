
import collections
import gzip
import json
import os
import random
import re

import enum
from bert import modeling
from bert import optimization
from bert import tokenization
import numpy as np
import tensorflow as tf
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import tpu as contrib_tpu


flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string(
  "tpu", default=None,
  help="The Cloud TPU to use for training. This should be either the name "
      "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
      "url.")

flags.DEFINE_string("train_precomputed_file", None,
                    "Precomputed tf records for training.")

flags.DEFINE_integer("train_num_precomputed", None,
                     "Number of precomputed tf records for training.")


flags.DEFINE_bool(
    "aoa", False,
    "whether to do attention-over-attention")

flags.DEFINE_bool(
    "freeze_bert", False,
    "whether to freeze bert base")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

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

# flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("answer_type_loss_coef", 1.0, "answer_type_loss_coef")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 500,
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

flags.DEFINE_float(
    "include_unknowns", -1.0,
    "If positive, probability of including answers of type `UNKNOWN`.")

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


flags.DEFINE_bool("do_train", True, "Whether to run eval on the dev set.")

flags.DEFINE_string(
    "bert_config_file", 'bert-joint-baseline/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", 'bert-joint-baseline/vocab-nq.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", 'bert_model_output',
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "model_suffix", '',
    "model_suffix")


flags.DEFINE_string(
    "init_checkpoint", 'bert-joint-baseline/bert_joint.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string('f', '', 'kernel')

TextSpan = collections.namedtuple("TextSpan", "token_positions text")


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu, freeze_bert=False):
  """Creates an optimizer training op."""
  global_step = tf.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=1.0,
      cycle=False)

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  optimizer = optimization.AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

  if use_tpu:
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

  tvars = tf.trainable_variables()
  # all pretrained weights inside BERT starts with 'bert'
  if freeze_bert:
    tvars = [tvar for tvar in tvars if not tvar.name.startswith('bert')]
  grads = tf.gradients(loss, tvars)

  # This is how the model was pre-trained.
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)

  # Normally the global step update is done inside of `apply_gradients`.
  # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
  # a different optimizer, you should probably take this line out.
  new_global_step = global_step + 1
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op



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


def has_long_answer(a):
  return (a["long_answer"]["start_token"] >= 0 and
          a["long_answer"]["end_token"] >= 0)


def should_skip_context(e, idx):
  if (FLAGS.skip_nested_contexts and
      not e["long_answer_candidates"][idx]["top_level"]):
    return True
  elif not get_candidate_text(e, idx).text.strip():
    # Skip empty contexts.
    return True
  else:
    return False


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
    context_type = get_candidate_type(e, idx)
    if counts[context_type] < FLAGS.max_position:
      counts[context_type] += 1
    c["type_and_position"] = "[%s=%d]" % (context_type, counts[context_type])


def get_candidate_type_and_position(e, idx):
  """Returns type and position info for the candidate at the given index."""
  if idx == -1:
    return "[NoLongAnswer]"
  else:
    return e["long_answer_candidates"][idx]["type_and_position"]


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


def create_example_from_jsonl_simple(line):
  """Creates an NQ example from a given line of JSON."""
  e = json.loads(line.decode('utf-8'), object_pairs_hook=collections.OrderedDict)
  document_tokens = e["document_text"].split(" ")
  e["document_tokens"] = []
  for token in document_tokens:
      e["document_tokens"].append({"token":token, "start_byte":-1, "end_byte":-1, "html_token":"<" in token})

  add_candidate_types_and_positions(e)
  try:
    annotation, annotated_idx, annotated_sa = get_first_annotation(e)
  except:
    annotation, annotated_idx, annotated_sa = None, -1, (-1, -1)  

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

  # Yes/no answers are added in the input text.
  if annotation is not None:
    assert annotation["yes_no_answer"] in ("YES", "NO", "NONE")
    if annotation["yes_no_answer"] in ("YES", "NO"):
      answer["input_text"] = annotation["yes_no_answer"].lower()

  # Add a short answer if one was found.
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
    assert expected_answer_text == answer["span_text"], (expected_answer_text,
                                                         answer["span_text"])

  # Add a long answer if one was found.
  elif annotation and annotation["long_answer"]["candidate_index"] >= 0:
    answer["span_text"] = get_candidate_text(e, annotated_idx).text
    answer["span_start"] = 0
    answer["span_end"] = len(answer["span_text"])

  context_idxs = [-1]
  context_list = [{"id": -1, "type": get_candidate_type_and_position(e, -1)}]
  context_list[-1]["text_map"], context_list[-1]["text"] = (
      get_candidate_text(e, -1))
  for idx, _ in candidates_iter(e):
    context = {"id": idx, "type": get_candidate_type_and_position(e, idx)}
    context["text_map"], context["text"] = get_candidate_text(e, idx)
    context_idxs.append(idx)
    context_list.append(context)
    if len(context_list) >= FLAGS.max_contexts:
      break

  if "document_title" not in e:
      e["document_title"] = e["example_id"]

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
  for context in context_list:
    single_map.extend([-1, -1])
    single_context.append("[ContextId=%d] %s" %
                          (context["id"], context["type"]))
    offset += len(single_context[-1]) + 1
    if context["id"] == annotated_idx:
      answer["span_start"] += offset
      answer["span_end"] += offset

    # Many contexts are empty once the HTML tags have been stripped, so we
    # want to skip those.
    if context["text"]:
      single_map.extend(context["text_map"])
      single_context.append(context["text"])
      offset += len(single_context[-1]) + 1

  example["contexts"] = " ".join(single_context)
  example["contexts_map"] = single_map
  if annotated_idx in context_idxs:
    expected = example["contexts"][answer["span_start"]:answer["span_end"]]

    # This is a sanity check to ensure that the calculated start and end
    # indices match the reported span text. If this assert fails, it is likely
    # a bug in the data preparation code above.
    assert expected == answer["span_text"], (expected, answer["span_text"])

  return example

def create_example_from_jsonl(line):
  """Creates an NQ example from a given line of JSON."""
  e = json.loads(line.decode('utf-8'), object_pairs_hook=collections.OrderedDict)
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

  # Yes/no answers are added in the input text.
  if annotation is not None:
    assert annotation["yes_no_answer"] in ("YES", "NO", "NONE")
    if annotation["yes_no_answer"] in ("YES", "NO"):
      answer["input_text"] = annotation["yes_no_answer"].lower()

  # Add a short answer if one was found.
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
    assert expected_answer_text == answer["span_text"], (expected_answer_text,
                                                         answer["span_text"])

  # Add a long answer if one was found.
  elif annotation and annotation["long_answer"]["candidate_index"] >= 0:
    answer["span_text"] = get_candidate_text(e, annotated_idx).text
    answer["span_start"] = 0
    answer["span_end"] = len(answer["span_text"])

  context_idxs = [-1]
  context_list = [{"id": -1, "type": get_candidate_type_and_position(e, -1)}]
  context_list[-1]["text_map"], context_list[-1]["text"] = (
      get_candidate_text(e, -1))
  for idx, _ in candidates_iter(e):
    context = {"id": idx, "type": get_candidate_type_and_position(e, idx)}
    context["text_map"], context["text"] = get_candidate_text(e, idx)
    context_idxs.append(idx)
    context_list.append(context)
    if len(context_list) >= FLAGS.max_contexts:
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
  for context in context_list:
    single_map.extend([-1, -1])
    single_context.append("[ContextId=%d] %s" %
                          (context["id"], context["type"]))
    offset += len(single_context[-1]) + 1
    if context["id"] == annotated_idx:
      answer["span_start"] += offset
      answer["span_end"] += offset

    # Many contexts are empty once the HTML tags have been stripped, so we
    # want to skip those.
    if context["text"]:
      single_map.extend(context["text_map"])
      single_context.append(context["text"])
      offset += len(single_context[-1]) + 1

  example["contexts"] = " ".join(single_context)
  example["contexts_map"] = single_map
  if annotated_idx in context_idxs:
    expected = example["contexts"][answer["span_start"]:answer["span_end"]]

    # This is a sanity check to ensure that the calculated start and end
    # indices match the reported span text. If this assert fails, it is likely
    # a bug in the data preparation code above.
    assert expected == answer["span_text"], (expected, answer["span_text"])

  return example


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


def convert_examples_to_features(examples, tokenizer, is_training, output_fn):
  """Converts a list of NqExamples into InputFeatures."""
  num_spans_to_ids = collections.defaultdict(list)

  for example in examples:
    example_index = example.example_id
    features = convert_single_example(example, tokenizer, is_training)
    num_spans_to_ids[len(features)].append(example.qas_id)

    for feature in features:
      feature.example_index = example_index
      feature.unique_id = feature.example_index + feature.doc_span_index
      output_fn(feature)

  return num_spans_to_ids


def convert_single_example(example, tokenizer, is_training):
  """Converts a single NqExample into a list of InputFeatures."""
  tok_to_orig_index = []
  orig_to_tok_index = []
  all_doc_tokens = []
  features = []
  for (i, token) in enumerate(example.doc_tokens):
    orig_to_tok_index.append(len(all_doc_tokens))
    sub_tokens = tokenize(tokenizer, token)
    tok_to_orig_index.extend([i] * len(sub_tokens))
    all_doc_tokens.extend(sub_tokens)

  # `tok_to_orig_index` maps wordpiece indices to indices of whitespace
  # tokenized word tokens in the contexts. The word tokens might themselves
  # correspond to word tokens in a larger document, with the mapping given
  # by `doc_tokens_map`.
  if example.doc_tokens_map:
    tok_to_orig_index = [
        example.doc_tokens_map[index] for index in tok_to_orig_index
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
      if ((not contains_an_annotation) or
          example.answer.type == AnswerType.UNKNOWN):
        # If an example has unknown answer type or does not contain the answer
        # span, then we only include it with probability --include_unknowns.
        # When we include an example with unknown answer type, we set the first
        # token of the passage to be the annotated short span.
        if (FLAGS.include_unknowns < 0 or
            random.random() > FLAGS.include_unknowns):
          continue
        start_position = 0
        end_position = 0
        answer_type = AnswerType.UNKNOWN
      else:
        doc_offset = len(query_tokens) + 2
        start_position = tok_start_position - doc_start + doc_offset
        end_position = tok_end_position - doc_start + doc_offset
        answer_type = example.answer.type

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


class CreateTFExampleFn(object):
  """Functor for creating NQ tf.Examples."""

  def __init__(self, is_training):
    self.is_training = is_training
    self.tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  def process(self, example):
    """Coverts an NQ example in a list of serialized tf examples."""
    nq_examples = read_nq_entry(example, self.is_training)
    input_features = []
    for nq_example in nq_examples:
      input_features.extend(
          convert_single_example(nq_example, self.tokenizer, self.is_training))

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
        for k, v in input_feature.token_to_orig_map.items():
          token_map[k] = v
        features["token_map"] = create_int_feature(token_map)

      yield tf.train.Example(features=tf.train.Features(
          feature=features)).SerializeToString()


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


def read_nq_examples(input_file, is_training):
  """Read a NQ json file into a list of NqExample."""
  input_paths = tf.gfile.Glob(input_file)
  input_data = []

  def _open(path):
    if path.endswith(".gz"):
      return gzip.GzipFile(fileobj=tf.gfile.Open(path, "r"))
    else:
      return tf.gfile.Open(path, "r")

  for path in input_paths:
    tf.logging.info("Reading: %s", path)
    with _open(path) as input_file:
      for line in input_file:
        try:
          input_data.append(create_example_from_jsonl(line))
        except:
          input_data.append(create_example_from_jsonl_simple(line))
        # if len(input_data)==1: break

  examples = []
  for entry in input_data:
    examples.extend(read_nq_entry(entry, is_training))
  return examples

# Softmax over axis
def softmax(target, axis, mask, epsilon=1e-12, name=None):
  with tf.op_scope([target], name, 'softmax'):
    max_axis = tf.reduce_max(target, axis, keep_dims=True)
    target_exp = tf.exp(target-max_axis) * mask
    normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
    softmax = target_exp / (normalize + epsilon)
    return softmax

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings, aoa=False):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # Get the logits for the start and end predictions.
  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]  # bs
  seq_length = final_hidden_shape[1]  # 512
  hidden_size = final_hidden_shape[2] # 1024

  output_weights = tf.get_variable(              # [2,1024], W_1 and W_2
      "cls/nq/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(     # [2]
      "cls/nq/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,    # [ bs * 512, 1024]
                                   [batch_size * seq_length, hidden_size])

  # W_1 * H^L and W_2 * H^L
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)  # [ bs * 512, 2]
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])  # [ bs, 512, 2]
  logits = tf.transpose(logits, [2, 0, 1])                  # [ 2, bs, 512]

  unstacked_logits = tf.unstack(logits, axis=0)   # [ [bs, 512], [bs, 512] ]

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  if aoa:
    len_q = 32

    query_mask = tf.to_float(input_mask) - tf.to_float(segment_ids)      # [bs,512]
    query_mask = query_mask[:, 1:(len_q+1)]  # skip [CLS]                # [bs, 32]

    doc_mask = tf.to_float(segment_ids)                                  # [bs,512]

    HQ = final_hidden[:,1:(len_q+1),:]               # [bs,M,1024]
    HC = final_hidden                                # [bs,N,1024]

    HQ_matrix = tf.reshape(HQ, [batch_size * len_q, hidden_size])                # [bs * M,1024]
    HC_matrix = tf.reshape(HC, [batch_size * seq_length, hidden_size])           # [bs * N,1024]

    wq1 = tf.get_variable(  # [1024,1024], W^Q_1
      "cls/nq/aoa_wq1", [hidden_size, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))
    wq2 = tf.get_variable(  # [1024,1024], W^Q_1
      "cls/nq/aoa_wq2", [hidden_size, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    wc1 = tf.get_variable(  # [1024,1024], W^Q_1
      "cls/nq/aoa_wc1", [hidden_size, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))
    wc2 = tf.get_variable(  # [1024,1024], W^Q_1
      "cls/nq/aoa_wc2", [hidden_size, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))
    
    hq1_bias = tf.get_variable("cls/nq/hq1_bias", [len_q], initializer=tf.zeros_initializer())
    hq2_bias = tf.get_variable("cls/nq/hq2_bias", [len_q], initializer=tf.zeros_initializer())
    hc1_bias = tf.get_variable("cls/nq/hc1_bias", [seq_length], initializer=tf.zeros_initializer())
    hc2_bias = tf.get_variable("cls/nq/hc2_bias", [seq_length], initializer=tf.zeros_initializer())

    HQ1 = tf.matmul(HQ_matrix, wq1)  # [bs * M, 1024]
    HQ1 = tf.reshape(HQ1, [batch_size, len_q, hidden_size])  # [bs, M, 1024]
    HQ1 = tf.add(HQ1, tf.expand_dims(tf.expand_dims(hq1_bias,0),-1))

    HQ2 = tf.matmul(HQ_matrix, wq2)  # [bs * M, 1024]
    HQ2 = tf.reshape(HQ2, [batch_size, len_q, hidden_size])  # [bs, M, 1024]
    HQ2 = tf.add(HQ2, tf.expand_dims(tf.expand_dims(hq2_bias,0),-1))

    HC1 = tf.matmul(HC_matrix, wc1)  # [bs * N, 1024]
    HC1 = tf.reshape(HC1, [batch_size, seq_length, hidden_size])  # [bs, N, 1024]
    HC1 = tf.add(HC1, tf.expand_dims(tf.expand_dims(hc1_bias,0),-1))

    HC2 = tf.matmul(HC_matrix, wc2)  # [bs * M, 1024]
    HC2 = tf.reshape(HC2, [batch_size, seq_length, hidden_size])  # [bs, N, 1024]
    HC2 = tf.add(HC2, tf.expand_dims(tf.expand_dims(hc2_bias,0),-1))    

    M1 = tf.matmul(HC1, HQ1, transpose_b=True)  # [bs, 512, 32]
    M2 = tf.matmul(HC2, HQ2, transpose_b=True)  # [bs, N, M]

    M_mask = tf.to_float(tf.matmul(tf.expand_dims(doc_mask, -1), tf.expand_dims(query_mask, 1))) # [bs, 512, 32]
    alpha1 = softmax(M1, 1, M_mask)     # [bs, N, M]
    beta1 = softmax(M1, 2, M_mask)      # [bs, N, M]
    alpha2 = softmax(M2, 1, M_mask)
    beta2 = softmax(M2, 2, M_mask)

    query_importance1 = tf.expand_dims(tf.reduce_sum(beta1, 1) / tf.to_float(tf.expand_dims(seq_length, -1)), -1) # [bs, M, 1]
    s1 = tf.squeeze(tf.matmul(alpha1, query_importance1), [2])  # [bs, N]
    query_importance2 = tf.expand_dims(tf.reduce_sum(beta2, 1) / tf.to_float(tf.expand_dims(seq_length, -1)), -1) # [bs, M, 1]
    s2 = tf.squeeze(tf.matmul(alpha2, query_importance2), [2])  # [bs, N]

    aoa_wts1 = tf.get_variable("cls/nq/aoa_and_logits_wts1", [2], initializer=tf.truncated_normal_initializer(stddev=0.02))  # [2]
    aoa_wts2 = tf.get_variable("cls/nq/aoa_and_logits_wts2", [2], initializer=tf.truncated_normal_initializer(stddev=0.02))  # [2]

    aoa_wts1 = tf.nn.softmax(aoa_wts1)
    aoa_wts2 = tf.nn.softmax(aoa_wts2)

    start_logits = aoa_wts1[0] * start_logits + aoa_wts1[1] * s1
    end_logits   = aoa_wts2[0] * end_logits   + aoa_wts2[1] * s2


  # Get the logits for the answer type prediction.
  answer_type_output_layer = model.get_pooled_output()
  answer_type_hidden_size = answer_type_output_layer.shape[-1].value

  num_answer_types = 5  # YES, NO, UNKNOWN, SHORT, LONG
  answer_type_output_weights = tf.get_variable(
      "answer_type_output_weights", [num_answer_types, answer_type_hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  answer_type_output_bias = tf.get_variable(
      "answer_type_output_bias", [num_answer_types],
      initializer=tf.zeros_initializer())

  answer_type_logits = tf.matmul(
      answer_type_output_layer, answer_type_output_weights, transpose_b=True)
  answer_type_logits = tf.nn.bias_add(answer_type_logits,
                                      answer_type_output_bias)

  return (start_logits, end_logits, answer_type_logits)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, aoa=False, freeze_bert=False):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (start_logits, end_logits, answer_type_logits) = create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        aoa=aoa)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      seq_length = modeling.get_shape_list(input_ids)[1]

      # Computes the loss for positions.
      def compute_loss(logits, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss

      # Computes the loss for labels.
      def compute_label_loss(logits, labels):
        one_hot_labels = tf.one_hot(
            labels, depth=len(AnswerType), dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_labels * log_probs, axis=-1))
        return loss

      start_positions = features["start_positions"]
      end_positions = features["end_positions"]
      answer_types = features["answer_types"]

      start_loss = compute_loss(start_logits, start_positions)
      end_loss = compute_loss(end_logits, end_positions)
      answer_type_loss = compute_label_loss(answer_type_logits, answer_types)

      total_loss = (start_loss + end_loss + answer_type_loss * FLAGS.answer_type_loss_coef) / 3.0

      train_op = create_optimizer(total_loss, learning_rate,
                                               num_train_steps,
                                               num_warmup_steps, use_tpu, freeze_bert=freeze_bert)

      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "unique_ids": unique_ids,
          "start_logits": start_logits,
          "end_logits": end_logits,
          "answer_type_logits": answer_type_logits,
      }
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and PREDICT modes are supported: %s" %
                       (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }

  if is_training:
    name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["answer_types"] = tf.FixedLenFeature([], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file, num_parallel_reads=50)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


RawResult = collections.namedtuple(
    "RawResult",
    ["unique_id", "start_logits", "end_logits", "answer_type_logits"])


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    if self.is_training:
      features["start_positions"] = create_int_feature([feature.start_position])
      features["end_positions"] = create_int_feature([feature.end_position])
      features["answer_types"] = create_int_feature([feature.answer_type])
    else:
      token_map = [-1] * len(feature.input_ids)
      for k, v in feature.token_to_orig_map.items():
        token_map[k] = v
      features["token_map"] = create_int_feature(token_map)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


Span = collections.namedtuple("Span", ["start_token_idx", "end_token_idx"])


class EvalExample(object):
  """Eval data available for a single example."""

  def __init__(self, example_id, candidates):
    self.example_id = example_id
    self.candidates = candidates
    self.results = {}
    self.features = {}


class ScoreSummary(object):

  def __init__(self):
    self.predicted_label = None
    self.short_span_score = None
    self.cls_token_score = None
    self.answer_type_logits = None


def read_candidates_from_one_split(input_path):
  
  candidates_dict = {}  
  try:    
    with gzip.GzipFile(fileobj=tf.gfile.Open(input_path)) as input_file:
      tf.logging.info("Reading examples from: %s", input_path)
      for line in input_file:
        e = json.loads(line.decode('utf-8'))
        candidates_dict[e["example_id"]] = e["long_answer_candidates"]
  except:
    with tf.io.gfile.GFile(input_path, "r") as input_file:
      tf.logging.info("Reading examples from: %s", input_path)
      for line in input_file:
        e = json.loads(line.decode('utf-8'))
        candidates_dict[e["example_id"]] = e["long_answer_candidates"]    

  return candidates_dict  


def read_candidates(input_pattern):
  """Read candidates with real multiple processes."""
  input_paths = tf.gfile.Glob(input_pattern)
  final_dict = {}
  for input_path in input_paths:
    final_dict.update(read_candidates_from_one_split(input_path))
  return final_dict


def get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(
      enumerate(logits[1:], 1), key=lambda x: x[1], reverse=True)
  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def compute_predictions(example):
  """Converts an example into an NQEval object for evaluation."""
  predictions = []
  n_best_size = 10
  max_answer_length = 30

  for unique_id, result in example.results.items():
    if unique_id not in example.features:
      raise ValueError("No feature found with unique_id:", unique_id)
    token_map = example.features[unique_id]["token_map"].int64_list.value
    start_indexes = get_best_indexes(result["start_logits"], n_best_size)
    end_indexes = get_best_indexes(result["end_logits"], n_best_size)
    for start_index in start_indexes:
      for end_index in end_indexes:
        if end_index < start_index:
          continue
        if token_map[start_index] == -1:
          continue
        if token_map[end_index] == -1:
          continue
        length = end_index - start_index + 1
        if length > max_answer_length:
          continue
        summary = ScoreSummary()
        summary.short_span_score = (
            result["start_logits"][start_index] +
            result["end_logits"][end_index])
        summary.cls_token_score = (
            result["start_logits"][0] + result["end_logits"][0])
        summary.answer_type_logits = result["answer_type_logits"]
        start_span = token_map[start_index]
        end_span = token_map[end_index] + 1

        # Span logits minus the cls logits seems to be close to the best.
        score = summary.short_span_score - summary.cls_token_score
        predictions.append((score, summary, start_span, end_span))

  # Default empty prediction.
  score = -10000.0
  short_span = Span(-1, -1)
  long_span = Span(-1, -1)
  summary = ScoreSummary()

  if predictions:
    score, summary, start_span, end_span = sorted(predictions, reverse=True)[0]
    short_span = Span(start_span, end_span)
    for c in example.candidates:
      start = short_span.start_token_idx
      end = short_span.end_token_idx
      if c["top_level"] and c["start_token"] <= start and c["end_token"] >= end:
        long_span = Span(c["start_token"], c["end_token"])
        break

  summary.predicted_label = {
      "example_id": example.example_id,
      "long_answer": {
          "start_token": long_span.start_token_idx,
          "end_token": long_span.end_token_idx,
          "start_byte": -1,
          "end_byte": -1
      },
      "long_answer_score": score,
      "short_answers": [{
          "start_token": short_span.start_token_idx,
          "end_token": short_span.end_token_idx,
          "start_byte": -1,
          "end_byte": -1
      }],
      "short_answers_score": score,
      "yes_no_answer": "NONE"
  }

  return summary


def compute_pred_dict(candidates_dict, dev_features, raw_results):
  """Computes official answer key from raw logits."""
  raw_results_by_id = [(int(res["unique_id"] + 1), res) for res in raw_results]

  # Cast example id to int32 for each example, similarly to the raw results.
  sess = tf.Session()
  all_candidates = candidates_dict.items()
  example_ids = tf.to_int32(np.array([int(k) for k, _ in all_candidates
                                     ])).eval(session=sess)
  examples_by_id = list(zip(example_ids, all_candidates))

  # Cast unique_id also to int32 for features.
  feature_ids = []
  features = []
  for f in dev_features:
    feature_ids.append(f.features.feature["unique_ids"].int64_list.value[0] + 1)
    features.append(f.features.feature)
  feature_ids = tf.to_int32(np.array(feature_ids)).eval(session=sess)
  features_by_id = list(zip(feature_ids, features))

  # Join examplew with features and raw results.
  examples = []
  merged = sorted(examples_by_id + raw_results_by_id + features_by_id)
  for idx, datum in merged:
    if isinstance(datum, tuple):
      examples.append(EvalExample(datum[0], datum[1]))
    elif "token_map" in datum:
      examples[-1].features[idx] = datum
    else:
      examples[-1].results[idx] = datum

  # Construct prediction objects.
  tf.logging.info("Computing predictions...")
  summary_dict = {}
  nq_pred_dict = {}
  for e in examples:
    summary = compute_predictions(e)
    summary_dict[e.example_id] = summary
    nq_pred_dict[e.example_id] = summary.predicted_label
    if len(nq_pred_dict) % 100 == 0:
      tf.logging.info("Examples processed: %d", len(nq_pred_dict))
  tf.logging.info("Done computing predictions.")

  return nq_pred_dict


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  if not FLAGS.do_train and not FLAGS.do_predict:
    raise ValueError("At least one of `{do_train,do_predict}` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_precomputed_file:
      raise ValueError("If `do_train` is True, then `train_precomputed_file` "
                       "must be specified.")
    if not FLAGS.train_num_precomputed:
      raise ValueError("If `do_train` is True, then `train_num_precomputed` "
                       "must be specified.")

  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))



def main_train(tpu_ip,aoa=False, freeze_bert=False):
    tf.logging.set_verbosity(tf.logging.INFO)
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    validate_flags_or_throw(bert_config)
    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    import json
    with tf.Session("grpc://"+tpu_ip+":8470") as sess:
      with open('/home/bo_liu/<your_credential_file>.json','r') as f:
                           auth_info = json.load(f)
      tf.contrib.cloud.configure_gcs(sess, credentials=auth_info)

    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project
    )   

    run_config = contrib_tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    #master=FLAGS.master,
    model_dir=FLAGS.output_dir,
    keep_checkpoint_max=100,
    session_config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True),
    save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    tpu_config=contrib_tpu.TPUConfig(
        iterations_per_loop=FLAGS.iterations_per_loop,
        num_shards=FLAGS.num_tpu_cores)) 

    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        num_train_features = FLAGS.train_num_precomputed
        num_train_steps = int(num_train_features / FLAGS.train_batch_size *
                              FLAGS.num_train_epochs)

        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
          bert_config=bert_config,
          init_checkpoint=FLAGS.init_checkpoint,
          learning_rate=FLAGS.learning_rate,
          num_train_steps=num_train_steps,
          num_warmup_steps=num_warmup_steps,
          use_tpu=FLAGS.use_tpu,
          use_one_hot_embeddings=FLAGS.use_tpu,
          aoa=aoa,
          freeze_bert=freeze_bert)

    # If TPU is not available, this falls back to normal Estimator on CPU or GPU.
    estimator = contrib_tpu.TPUEstimator(
          use_tpu=FLAGS.use_tpu,
          model_fn=model_fn,
          config=run_config,
          train_batch_size=FLAGS.train_batch_size,
          predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        tf.logging.info("***** Running training on precomputed features *****")
        tf.logging.info("  Num split examples = %d", num_train_features)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_filenames = tf.gfile.Glob(FLAGS.train_precomputed_file)
        train_input_fn = input_fn_builder(
            input_file=train_filenames,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)


def calc_num_examples(train_feature_file):
    train_filenames = tf.gfile.Glob(train_feature_file)
    assert len(train_filenames) == 50 or train_feature_file=="gs://bert-nq/bert-joint-baseline/nq-train.tfrecords-00000-of-00001"   
    return sum(sum(1 for _ in tf.python_io.tf_record_iterator(f)) for f in train_filenames)

def main(_):

    path = '/home/bo_liu/'

    FLAGS.use_tpu = True

    if FLAGS.tpu=='node3': tpu_ip = '10.75.44.26'
    elif FLAGS.tpu=='node4': tpu_ip = '10.218.168.66'
    elif FLAGS.tpu=='node2': tpu_ip = '10.240.1.2'
    elif FLAGS.tpu=='node-1': tpu_ip = '10.137.110.66'
    elif FLAGS.tpu=='node5': tpu_ip = '10.97.58.98'

    FLAGS.train_num_precomputed = calc_num_examples(FLAGS.train_precomputed_file)
    print('---------------------- FLAGS.train_num_precomputed = ', FLAGS.train_num_precomputed)    
    
    FLAGS.max_seq_length=512 
    FLAGS.save_checkpoints_steps=500   
    FLAGS.iterations_per_loop=500

    FLAGS.do_train=True
    FLAGS.output_dir='gs://tf2qa/bert_output'+FLAGS.model_suffix+'-{:d}-{:.2E}/'.format(FLAGS.train_batch_size, FLAGS.learning_rate)

    if FLAGS.do_lower_case:
        FLAGS.vocab_file = path+'vocab-nq.txt' 
        FLAGS.bert_config_file = path+'bert_config.json'
    else:
        FLAGS.vocab_file = path+'vocab_cased-nq.txt'  
        FLAGS.bert_config_file= path+'bert_config_cased.json'

    main_train(tpu_ip,aoa=FLAGS.aoa,freeze_bert=FLAGS.freeze_bert)


if __name__ == "__main__":
  tf.app.run()



