import torch
import time
import statistics
from transformers import BertTokenizer, BertModel
from torch.cuda import empty_cache

class SpeedBenchmark:
    def __init__(self, model_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def _clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def warm_up(self, input_text, num_warmup=10):
        """Warm up the model to ensure stable measurements"""
        print("Warming up...")
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(**inputs)
        self._clear_cache()
        
    def benchmark_latency(self, input_text, num_runs=100, batch_size=1, sequence_length=128):
        """Measure inference latency"""
        # Prepare input
        inputs = self.tokenizer(
            input_text, 
            padding='max_length',
            max_length=sequence_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Duplicate for batch size
        inputs = {k: v.repeat(batch_size, 1) for k, v in inputs.items()}
        
        # Measure latency
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = self.model(**inputs)
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds
                
        return {
            'mean_latency': statistics.mean(latencies),
            'median_latency': statistics.median(latencies),
            'std_dev': statistics.stdev(latencies),
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'p95_latency': sorted(latencies)[int(0.95 * len(latencies))],
            'throughput': (batch_size * 1000) / statistics.mean(latencies)  # samples per second
        }

def run_speed_benchmark():
    # Test configurations
    configs = [
        {'batch_size': 1, 'sequence_length': 128},
    ]
    
    # Sample text
    sample_text = "This is a test sentence for benchmarking BERT model performance."
    
    # Initialize benchmark
    benchmark = SpeedBenchmark()
    
    # Warm up
    benchmark.warm_up(sample_text)
    
    # Run benchmarks for each configuration
    print(f"\nRunning on: {benchmark.device}")
    print("-" * 80)
    
    for config in configs:
        print(f"\nBenchmarking with batch_size={config['batch_size']}, "
              f"sequence_length={config['sequence_length']}")
        
        results = benchmark.benchmark_latency(
            sample_text,
            batch_size=config['batch_size'],
            sequence_length=config['sequence_length']
        )
        
        print(f"Mean latency: {results['mean_latency']:.2f} ms")
        print(f"Median latency: {results['median_latency']:.2f} ms")
        print(f"P95 latency: {results['p95_latency']:.2f} ms")
        print(f"Throughput: {results['throughput']:.2f} samples/second")
        print(f"Standard deviation: {results['std_dev']:.2f} ms")
        print("-" * 80)

if __name__ == "__main__":
    run_speed_benchmark()

