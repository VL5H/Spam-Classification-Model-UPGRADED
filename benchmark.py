import time
import psutil
import os
from inference_engine import SMSClassifier
from logging_system import logger
import joblib

class ModelBenchmark:
    """Comprehensive benchmarking system for SMS spam detection models."""

    def __init__(self):
        self.classifier = SMSClassifier()

    def run_full_benchmark(self):
        """Run complete performance benchmark."""
        if not self.classifier.model:
            print("No model loaded for benchmarking.")
            return None

        print("Running comprehensive model benchmark...")
        print("=" * 60)

        # Test messages for benchmarking
        test_messages = [
            # Ham messages
            "Hey, how are you doing?",
            "Meeting scheduled for tomorrow at 3 PM",
            "Thanks for your help with the project",
            "Hi mom, I'll be home late tonight",
            "Can we reschedule our call?",
            "Please review the attached document",
            "The weather is nice today",
            "Don't forget to buy groceries",
            "Happy birthday! Hope you have a great day",
            "Let me know if you need any assistance",

            # Spam messages
            "WINNER! You have won a £1000 cash prize! Call now!",
            "URGENT: Your account needs verification. Click this link immediately!",
            "FREE entry into our competition. Text YES to enter!",
            "Congratulations! You've been selected for a special offer!",
            "ALERT: Suspicious activity detected on your account",
            "You have won a free iPhone! Click here to claim your prize!",
            "Limited time offer: 90% discount on all products!",
            "Your package is waiting for pickup. Confirm delivery now!",
            "Investment opportunity: Double your money in 24 hours!",
            "Medical alert: Your prescription is ready for pickup"
        ]

        # Initialize metrics
        results = []
        total_start_time = time.time()

        print("Testing individual classifications...")
        for i, msg in enumerate(test_messages, 1):
            msg_start = time.time()
            result = self.classifier.classify_message(msg)
            msg_end = time.time()

            if "error" not in result:
                results.append({
                    'message': msg,
                    'classification': result['classification'],
                    'confidence': result['confidence'],
                    'time': msg_end - msg_start,
                    'expected': 'spam' if i > 10 else 'ham'  # First 10 ham, last 10 spam
                })
                print(f"  Message {i}: {result['classification']} (confidence: {result['confidence']:.3f})")
            else:
                print(f"  Message {i}: ERROR - {result['error']}")

        total_time = time.time() - total_start_time

        if not results:
            print("Benchmark failed - no valid results.")
            return None

        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(results, total_time)

        # Memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        metrics['memory_usage_mb'] = memory_mb

        # Model info
        model_info = self.classifier.get_model_info()
        metrics['model_info'] = model_info

        # Display results
        self._display_results(metrics)

        # Log benchmark completion
        logger.log_model_action("BENCHMARK_COMPLETED",
                                f"Total time: {total_time:.3f}s")

        return metrics

    def _calculate_metrics(self, results, total_time):
        """Calculate detailed performance metrics."""
        times = [r['time'] for r in results]
        confidences = [r['confidence'] for r in results]

        # Timing metrics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        median_time = sorted(times)[len(times) // 2]

        # Confidence metrics
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)

        # Accuracy metrics (based on expected classifications)
        correct_predictions = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for result in results:
            predicted = result['classification']
            expected = result['expected']
            is_correct = predicted == expected

            if is_correct:
                correct_predictions += 1
                if predicted == 'spam':
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if predicted == 'spam':
                    false_positives += 1
                else:
                    false_negatives += 1

        accuracy = correct_predictions / len(results)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Classification distribution
        spam_count = sum(1 for r in results if r['classification'] == 'spam')
        ham_count = len(results) - spam_count

        return {
            'total_messages': len(results),
            'total_time': total_time,
            'avg_time_per_message': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'median_time': median_time,
            'avg_confidence': avg_confidence,
            'min_confidence': min_confidence,
            'max_confidence': max_confidence,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'spam_detections': spam_count,
            'ham_detections': ham_count,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }

    def _display_results(self, metrics):
        """Display benchmark results in a formatted way."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)

        print("PERFORMANCE METRICS:")
        print("-" * 30)
        print(f"Total messages tested: {metrics['total_messages']}")
        print(f"Total time: {metrics['total_time']:.1f}s")
        print(f"Average time per message: {metrics['avg_time_per_message']:.4f}s")
        print(f"Min time: {metrics['min_time']:.4f}s")
        print(f"Max time: {metrics['max_time']:.4f}s")
        print(f"Median time: {metrics['median_time']:.4f}s")

        print("\nTIMING ANALYSIS:")
        print("-" * 30)
        print(f"Average time per message: {metrics['avg_time_per_message']:.4f}s")
        print(f"Min time: {metrics['min_time']:.4f}s")
        print(f"Max time: {metrics['max_time']:.4f}s")
        print(f"Median time: {metrics['median_time']:.4f}s")

        print("\nCONFIDENCE ANALYSIS:")
        print("-" * 30)
        print(f"Average confidence: {metrics['avg_confidence']:.3f}")
        print(f"Min confidence: {metrics['min_confidence']:.3f}")
        print(f"Max confidence: {metrics['max_confidence']:.3f}")

        print("\nACCURACY METRICS:")
        print("-" * 30)
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1 Score: {metrics['f1_score']:.3f}")

        print("\nCLASSIFICATION DISTRIBUTION:")
        print("-" * 30)
        print(f"Spam detections: {metrics['spam_detections']}")
        print(f"Ham detections: {metrics['ham_detections']}")
        print(f"True positives: {metrics['true_positives']}")
        print(f"False positives: {metrics['false_positives']}")
        print(f"True negatives: {metrics['true_negatives']}")
        print(f"False negatives: {metrics['false_negatives']}")

        print("\nRESOURCE USAGE:")
        print("-" * 30)
        print(f"Memory usage: {metrics['memory_usage_mb']:.1f} MB")

        if 'model_info' in metrics and metrics['model_info']:
            print("\nMODEL INFORMATION:")
            print("-" * 30)
            model_info = metrics['model_info']
            if 'model_name' in model_info:
                print(f"Model type: {model_info['model_name']}")
            if 'metrics' in model_info and model_info['metrics']:
                m = model_info['metrics']
                print(f"Accuracy: {m.get('accuracy', 0):.3f}")

        print("=" * 60)

    def benchmark_model_loading(self):
        """Benchmark model loading time."""
        print("Benchmarking model loading time...")

        model_files = [f for f in os.listdir('.') if f.startswith(('SVM_', 'LogisticRegression_', 'RandomForest_')) and f.endswith('.pkl')]

        if not model_files:
            print("No model files found for loading benchmark.")
            return

        loading_times = []

        for model_file in model_files:
            start_time = time.time()

            try:
                # Load model
                model = joblib.load(model_file)
                end_time = time.time()
                loading_time = end_time - start_time
                loading_times.append(loading_time)
                print(f"  {model_file}: {loading_time:.3f}s")
            except Exception as e:
                print(f"  {model_file}: ERROR - {str(e)}")

        if loading_times:
            avg_loading_time = sum(loading_times) / len(loading_times)
            print(f"Average loading time: {avg_loading_time:.3f}s")
            print(f"Min loading time: {min(loading_times):.3f}s")
            print(f"Max loading time: {max(loading_times):.3f}s")

    def benchmark_memory_usage(self):
        """Benchmark memory usage during different operations."""
        print("Benchmarking memory usage...")

        process = psutil.Process()

        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024
        print(f"Baseline memory: {baseline_memory:.1f} MB")

        # Memory after loading model
        if self.classifier.model:
            model_loaded_memory = process.memory_info().rss / 1024 / 1024
            model_memory_delta = model_loaded_memory - baseline_memory
            print(f"Memory after model load: {model_loaded_memory:.1f} MB")

        # Memory during classification
        test_message = "This is a test message for memory benchmarking."
        start_memory = process.memory_info().rss / 1024 / 1024

        result = self.classifier.classify_message(test_message)

        end_memory = process.memory_info().rss / 1024 / 1024
        classification_memory_delta = end_memory - start_memory

        print(f"Memory delta during classification: {classification_memory_delta:.1f} MB")

        return {
            'baseline_memory': baseline_memory,
            'model_memory_delta': model_memory_delta if 'model_memory_delta' in locals() else 0,
            'classification_memory_delta': classification_memory_delta
        }

    def run_mobile_simulation(self):
        """Simulate mobile device constraints and performance."""
        print("Running mobile device simulation...")
        print("This simulates performance on a typical Android device.")

        # Simulate limited CPU by adding delays
        # (In real mobile deployment, this would be actual device constraints)

        if not self.classifier.model:
            print("No model loaded for mobile simulation.")
            return

        # Test with smaller batch to simulate mobile usage patterns
        mobile_messages = [
            "Hey, how are you?",
            "WINNER! Claim your prize now!",
            "Meeting at 3 PM",
            "URGENT: Account verification needed",
            "Thanks for your help"
        ]

        print("Simulating mobile classification performance...")

        results = []
        for msg in mobile_messages:
            # Add small delay to simulate mobile CPU constraints
            time.sleep(0.01)

            start_time = time.time()
            result = self.classifier.classify_message(msg)
            end_time = time.time()

            if "error" not in result:
                results.append({
                    'time': end_time - start_time,
                    'confidence': result['confidence']
                })

        if results:
            avg_mobile_time = sum(r['time'] for r in results) / len(results)
            avg_mobile_confidence = sum(r['confidence'] for r in results) / len(results)

            print("Mobile Simulation Results:")
            print(f"  Average time: {avg_mobile_time:.4f}s")
            print(f"  Average confidence: {avg_mobile_confidence:.3f}")
            print("  (Target: < 100ms per message for good mobile UX)")

            if avg_mobile_time < 0.1:  # 100ms
                print("  ✓ Performance suitable for mobile devices")
            else:
                print("  ⚠ Performance may be slow on low-end mobile devices")

def main():
    """Command-line interface for benchmarking."""
    benchmark = ModelBenchmark()

    while True:
        print("\n" + "="*50)
        print("MODEL BENCHMARKING SYSTEM")
        print("="*50)
        print("1. Run full performance benchmark")
        print("2. Benchmark model loading time")
        print("3. Benchmark memory usage")
        print("4. Run mobile device simulation")
        print("5. Back to main menu")

        choice = input("Enter your choice (1-5): ").strip()

        if choice == '1':
            benchmark.run_full_benchmark()
        elif choice == '2':
            benchmark.benchmark_model_loading()
        elif choice == '3':
            benchmark.benchmark_memory_usage()
        elif choice == '4':
            benchmark.run_mobile_simulation()
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please enter a number between 1-5.")

if __name__ == "__main__":
    main()