"""
evaluation -- measurements taken on a TRAINED model, separate from the stages that train it.

    python -m evaluation.content_extrapolation      perplexity at 1x..128x the training length

A stage package (pretrain/, sft/, cot/, ...) owns an algorithm that produces a checkpoint.
Everything here owns a QUESTION asked of a checkpoint that already exists: nothing in this
package writes weights, and each module can be run against any stage's output without the
pipeline being involved. Run as modules (`python -m evaluation.<name>`) rather than as scripts,
so the project root is on the import path and `import default_config` resolves.
"""
