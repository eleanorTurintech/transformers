import torch
import time
import statistics
from transformers import BertTokenizer, BertModel
import gc
import numpy as np

class SpeedBenchmark:
    def __init__(self, model_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _clear_cache(self):
        """Thoroughly clear all caches"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _cooldown(self, seconds=0.5):
        """Add a cooldown period between runs"""
        time.sleep(seconds)

    def warm_up(self, input_text, num_warmup=10):
        """Warm up the model with proper cache clearing"""
        print("Warming up...")
        self._clear_cache()
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(**inputs)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        self._clear_cache()
        self._cooldown()

    def benchmark_latency(self, input_text, num_runs=100, batch_size=1, sequence_length=128):
        """Measure inference latency with consistent timing"""
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
            for run in range(num_runs):
                # Clear cache every few runs to prevent optimization buildup
                if run % 10 == 0:
                    self._clear_cache()
                    self._cooldown(0.1)

                # Ensure all previous CUDA operations are finished
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                _ = self.model(**inputs)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds

        # Remove outliers (optional)
        latencies = np.array(latencies)
        q1 = np.percentile(latencies, 25)
        q3 = np.percentile(latencies, 75)
        iqr = q3 - q1
        latencies = latencies[(latencies >= q1 - 1.5 * iqr) & (latencies <= q3 + 1.5 * iqr)]
                
        return {
            'mean_latency': statistics.mean(latencies),
            'median_latency': statistics.median(latencies),
            'std_dev': statistics.stdev(latencies),
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'p95_latency': sorted(latencies)[int(0.95 * len(latencies))],
            'throughput': (batch_size * 1000) / statistics.mean(latencies)
        }

def run_speed_benchmark(runs_per_config=100):
    # Test configurations
    configs = [
        {'batch_size': 1, 'sequence_length': 128},
        {'batch_size': 8, 'sequence_length': 128},
        {'batch_size': 32, 'sequence_length': 128},
    ]
    
    sample_text = "This is a test sentence for benchmarking BERT model performance."
    
    # Initialize benchmark
    benchmark = SpeedBenchmark()
    
    # Warm up
    benchmark.warm_up(sample_text)
    
    print(f"\nRunning on: {benchmark.device}")
    print("-" * 80)
    
    for config in configs:
        # Clear everything before each configuration
        benchmark._clear_cache()
        benchmark._cooldown(1.0)  # Longer cooldown between configs
        
        print(f"\nBenchmarking with batch_size={config['batch_size']}, "
              f"sequence_length={config['sequence_length']}")
        
        results = benchmark.benchmark_latency(
            sample_text,
            num_runs=runs_per_config,
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
