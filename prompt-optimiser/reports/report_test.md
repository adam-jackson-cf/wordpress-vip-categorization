# DSPy Optimization Report

**Generated:** 2025-11-16 15:54:47

## Summary

- **Optimizer:** bootstrap
- **Training Examples:** 12
- **Validation Examples:** 48
- **Validation Score:** 29.170
- **Duration:** 0.3 seconds

## Configuration

### Optimizer Settings
- **optimizer:** bootstrap
- **train_split:** 0.2
- **num_threads:** 1
- **seed:** 42
- **max_bootstrapped_demos:** 4
- **max_labeled_demos:** 8

### Model Settings
- **LLM Model:** gpt-4o-mini
- **LLM Base URL:** https://api.openai.com/v1
- **Temperature:** 0.3 (main), 1.0 (reflection for GEPA)

## Before Optimization

### Prompt Instructions
*Default DSPy ChainOfThought instructions*

- **Number of Demonstrations:** 0

## After Optimization

### Prompt Instructions
*Optimized DSPy ChainOfThought instructions*

- **Number of Demonstrations:** 0

## Changes

- **Instructions:** Changes may be in demonstrations or internal structure

## Notes

- DSPy optimizers may modify instructions, add demonstrations, or adjust internal parameters
- The optimized model is saved and can be loaded for production use
- Validation score indicates performance on held-out data