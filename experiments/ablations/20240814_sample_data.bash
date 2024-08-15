# Without wrappers
# Benchmarking
chemnlp-sample data/tabular/lipophilicity sampled_benchmark/ --benchmarking --class_balanced
chemnlp-sample data/tabular/bicerano_dataset sampled_benchmark/ --benchmarking --class_balanced
chemnlp-sample data/tabular/opv sampled_benchmark/ --benchmarking --class_balanced
chemnlp-sample data/tabular/melting_points sampled_benchmark/ --benchmarking --class_balanced
chemnlp-sample data/tabular/bc5disease sampled_benchmark/ --benchmarking --class_balanced
chemnlp-sample data/tabular/MUV_846 sampled_benchmark/ --benchmarking --class_balanced

# Default
chemnlp-sample data/tabular/lipophilicity/ sampled --class_balanced
chemnlp-sample data/tabular/bicerano_dataset/ sampled --class_balanced
chemnlp-sample data/tabular/opv/ sampled --class_balanced
chemnlp-sample data/tabular/melting_points/ sampled --class_balanced
chemnlp-sample data/tabular/bc5disease/ sampled --class_balanced
chemnlp-sample data/tabular/MUV_846/ sampled --class_balanced


# With wrappers
# Benchmarking
chemnlp-sample data/tabular/lipophilicity/ sampled_benchmark_wrapped/ --benchmarking --wrap-identifiers --class_balanced
chemnlp-sample data/tabular/bicerano_dataset/ sampled_benchmark_wrapped/ --benchmarking --wrap-identifiers --class_balanced
chemnlp-sample data/tabular/opv/ sampled_benchmark_wrapped/ --benchmarking --wrap-identifiers --class_balanced
chemnlp-sample data/tabular/melting_points/ sampled_benchmark_wrapped/ --benchmarking --wrap-identifiers --class_balanced
chemnlp-sample data/tabular/bc5disease/ sampled_benchmark_wrapped/ --benchmarking --wrap-identifiers --class_balanced
chemnlp-sample data/tabular/MUV_846/ sampled_benchmark_wrapped/ --benchmarking --wrap-identifiers --class_balanced

# Default
chemnlp-sample data/tabular/lipophilicity/ sampled_wrapped --wrap-identifiers --class_balanced
chemnlp-sample data/tabular/bicerano_dataset/ sampled_wrapped --wrap-identifiers --class_balanced
chemnlp-sample data/tabular/opv/ sampled_wrapped --wrap-identifiers --class_balanced
chemnlp-sample data/tabular/melting_points/ sampled_wrapped --wrap-identifiers --class_balanced
chemnlp-sample data/tabular/bc5disease/ sampled_wrapped --wrap-identifiers --class_balanced
chemnlp-sample data/tabular/MUV_846/ sampled_wrapped --wrap-identifiers --class_balanced
