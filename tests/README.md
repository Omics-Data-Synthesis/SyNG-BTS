# SyNG-BTS Test Suite

This directory contains the pytest test suite for the SyNG-BTS package.

## Test Structure

```
tests/
├── README.md              # This file
├── __init__.py            # Package marker
├── conftest.py            # Shared pytest fixtures
├── test_imports.py        # Package import and public API tests
├── test_data_utils.py     # Data loading, bundled datasets, path utilities
├── test_core.py           # Main experiment functions (generate, pilot_study, transfer)
├── test_evaluations.py    # Visualization functions (heatmap_eval, UMAP_eval, evaluation)
├── test_helpers.py        # Internal helper utilities and preprocessing
├── test_models.py         # Forward-pass tests for model architectures
├── test_training.py       # Training orchestrators (training_AEs, training_GANs, etc)
└── test_result.py         # SyngResult and PilotResult classes
```

## Running Tests

### Basic Usage

```bash
# Run all tests (281 pass, 1 skip)
pytest tests/ -v

# Run fast tests only (skip slow integration tests)
pytest tests/ -m "not slow"

# Run only slow tests
pytest tests/ -m slow

# Run tests with coverage
pytest tests/ --cov=syng_bts --cov-report=html

# Run specific test file
pytest tests/test_core.py -v

# Run specific test class
pytest tests/test_core.py::TestGenerate -v

# Run specific test
pytest tests/test_core.py::TestGenerate::test_returns_syng_result -v
```

### Debug Options

```bash
# Show print statements (useful for debugging)
pytest tests/ -v -s

# Stop on first failure
pytest tests/ -x

# Show full diff on assertion failures
pytest tests/ -v --tb=long

# Run with coverage and HTML report
pytest tests/ --cov=syng_bts --cov-report=html
# Then open htmlcov/index.html in browser
```

## Test Categories

- **Import Tests** (`test_imports.py`): Verifies the public API (package imports, exported functions `generate`, `pilot_study`, `transfer`), result objects, and that legacy names raise `ImportError`.
- **Data Utils Tests** (`test_data_utils.py`): Data loading and path utilities, bundled dataset operations, `resolve_data()` edge cases, and output-directory behavior.
- **Experiment Tests** (`test_core.py`): Main public API (`generate`, `pilot_study`, `transfer`) across input types and parameters; includes internal helper checks (`_parse_model_spec`, `_build_loss_df`, `_compute_new_size`) and slow integration tests.
- **Evaluation Tests** (`test_evaluations.py`): Visualization functions (`heatmap_eval`, `UMAP_eval`, `evaluation`) - should return figures (no `plt.show()`) and handle parameters correctly.
- **Helper Tests** (`test_helpers.py`): Internal utilities (preprocessing, seed management, label creation, augmentation) and pure functions (`generate_samples`, `reconstruct_samples`).
- **Model Tests** (`test_models.py`): Forward-pass checks for architectures (AE, VAE, CVAE, GAN) to validate tensor shapes and encoding behavior.
- **Result Tests** (`test_result.py`): `SyngResult` and `PilotResult` - construction, string representations, loss plotting, heatmap visualization, and CSV save.
<!-- - **Training Tests** (`test_training.py`): Training orchestrators (`training_AEs`, `training_GANs`, `training_flows`, `training_iter`) — return-type validation, loss computation, and removal of legacy I/O parameters. -->


## Fixtures

Common fixtures available in all test modules (defined in `conftest.py`):

| Fixture | Description |
|---------|-------------|
| `temp_dir` | Temporary directory with automatic cleanup |
| `sample_data` | Small 20×50 DataFrame mimicking transcriptomics data |
| `sample_data_with_labels` | Sample data with binary class labels column |
| `sample_csv_file` | Temporary CSV file for I/O testing |
| `small_training_config` | Minimal training config (2 epochs, lr=0.001) for fast tests |
| `sample_result` | Minimal SyngResult with generated data and loss (no model state) |
| `sample_result_with_model` | Full SyngResult including reconstructed data and model state |

## Adding New Tests

### Guidelines

1. **File organization**: Group tests by module/feature. For example, new functionality in `core.py` should go in `test_core.py`.
   
2. **Naming conventions**:
   - File: `test_*.py`
   - Class: `Test<Feature>` (e.g., `TestGenerate`)
   - Method: `test_<specific_case>` (e.g., `test_returns_syng_result`)
   
3. **Use fixtures** from `conftest.py` for common setup

4. **No I/O or side effects**: Training functions should not write files unless `output_dir` is explicitly provided

5. **No matplotlib**:  
   - Training helpers should not create figures
   - Visualization functions should return figures, not call `plt.show()`
   
6. **Test isolation**: Each test should be independent (no shared state)

## Best Practices

### Do
- Use fixtures for common setup (sample data, temp directories)
- Mark slow tests with `@pytest.mark.slow`
- Test both success and error cases
- Use *descriptive test names* that explain what's being tested
- Assert specific values, not just truthiness
- Group related tests in a class

### Don't
- Write tests with side effects (file creation without cleanup)
- Skip tests with `pytest.skip()` based on exception results
- Call `plt.show()` or `plt.savefig()` in code being tested
- Create module-level state that persists between tests
- Use hardcoded paths (use `temp_dir` fixture instead)
- Test implementation details instead of behavior