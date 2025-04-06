# PA1 - Locality Sensitive Hashing (LSH) for Genre Classification

## Authors
Tobias Winter, Nikolaus Reichsöllner

## Requirements

Put the FMA data in the `data` directory with the name `fma_metadata`.

This project uses Python 3 and the following libraries:

- numpy
- pandas
- scikit-learn
- scipy

Enable or disable run modes by changing these flags in the `main()` call:

```python
run_validation = True        # Run validation experiments with multiple configs
run_on_real_date = True      # Run final evaluation on test set
run_estimate = True          # Estimate runtime of exact nearest neighbor search
````
