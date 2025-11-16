# DSPy Optimization Report

**Generated:** 2025-11-16 15:54:59

## Summary

- **Optimizer:** gepa
- **Training Examples:** 72
- **Validation Examples:** 288
- **Validation Score:** 46.880
- **Duration:** 4.9 seconds

## Configuration

### Optimizer Settings
- **optimizer:** gepa
- **budget:** heavy
- **train_split:** 0.2
- **num_threads:** 4
- **seed:** 42

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