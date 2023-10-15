import os
import logging
import argparse

from .builtin_generators import resolve_summarizer_class

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="sample_inputs")
    parser.add_argument("--type", default="t5_cond")
    parser.add_argument("--model_path", default="NlpHUST/t5-small-vi-summarization")
    args = parser.parse_args()

    model_class = resolve_summarizer_class(args.type)
    summarizer = model_class(args.model_path)

    files = os.listdir(args.input_path)
    files = [os.path.join(args.input_path, f) for f in files]
    files = sorted(files)
    texts = []
    for f in files:
        with open(f) as reader:
            input_text = reader.read()
        texts.append(input_text)
    outputs = summarizer.greedy(texts)
    print(outputs)


if __name__ == "__main__":
    main()