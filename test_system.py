#!/usr/bin/env python3
"""
System Testing and End-to-End Workflow Test
Comprehensive testing of the SMS spam detection system.
"""

import os
import sys
import time
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from inference_engine import SMSClassifier
from feedback_system import FeedbackSystem
from logging_system import logger, echo_logger
from benchmark import ModelBenchmark

class SystemTester:
    """Comprehensive system testing and validation."""

    def __init__(self):
        self.classifier = SMSClassifier()
        self.feedback_system = FeedbackSystem()
        self.benchmark = ModelBenchmark()

    def run_full_system_test(self):
        """Run complete end-to-end system test."""
        print("=" * 80)
        print("SMS SPAM DETECTION SYSTEM - END-TO-END TEST")
        print("=" * 80)

        test_results = {
            'model_loading': False,
            'basic_classification': False,
            'confidence_scoring': False,
            'feedback_system': False,
            'logging_system': False,
            'echo_logging_system': False,
            'android_bridge': False,
            'android_echo_bridge': False,
            'performance_test': False,
            'memory_test': False,
            'gui_functionality': False
        }

        # Test 1: Model Loading
        print("\n1. Testing Model Loading...")
        if self.classifier.model and self.classifier.vectorizer:
            print("   [PASS] Model and vectorizer loaded successfully")
            test_results['model_loading'] = True
        else:
            print("   [FAIL] Failed to load model or vectorizer")
            return test_results

        # Test 2: Basic Classification
        print("\n2. Testing Basic Classification...")
        test_messages = [
            ("Hey, how are you doing?", "ham"),
            ("WINNER! You have won a free iPhone!", "spam"),
            ("Meeting scheduled for tomorrow", "ham"),
            ("URGENT: Your account is suspended", "spam")
        ]

        correct_predictions = 0
        for msg, expected in test_messages:
            result = self.classifier.classify_message(msg)
            if "error" not in result:
                predicted = result['classification']
                if predicted == expected:
                    correct_predictions += 1
                    print(f"   [PASS] '{msg[:30]}...' -> {predicted}")
                else:
                    print(f"   [FAIL] '{msg[:30]}...' -> {predicted} (expected {expected})")
            else:
                print(f"   [FAIL] Error classifying: {result['error']}")

        if correct_predictions >= 3:  # At least 75% accuracy
            test_results['basic_classification'] = True
            print(f"   Basic classification: {correct_predictions}/{len(test_messages)} correct")
        else:
            print(f"   Basic classification failed: {correct_predictions}/{len(test_messages)} correct")

        # Test 3: Confidence Scoring
        print("\n3. Testing Confidence Scoring...")
        confidence_test_msg = "Congratulations! You've won a lottery!"
        result = self.classifier.classify_message(confidence_test_msg)

        if "error" not in result and 'confidence' in result:
            confidence = result['confidence']
            needs_feedback = result.get('needs_feedback', False)
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Needs feedback: {needs_feedback}")

            if 0.0 <= confidence <= 1.0:
                test_results['confidence_scoring'] = True
                print("   [PASS] Confidence scoring working correctly")
            else:
                print("   [FAIL] Invalid confidence value")
        else:
            print("   [FAIL] Confidence scoring failed")

        # Test 4: Feedback System
        print("\n4. Testing Feedback System...")
        try:
            # Add test feedback
            self.feedback_system.log_feedback(
                "Test message for feedback",
                "spam",
                "spam",
                0.95
            )

            # Check if feedback was logged
            feedback_data = self.feedback_system.load_feedback_data()
            if len(feedback_data) > 0:
                print(f"   [PASS] Feedback system working ({len(feedback_data)} entries)")
                test_results['feedback_system'] = True
            else:
                print("   [FAIL] Feedback not recorded")
        except Exception as e:
            print(f"   [FAIL] Feedback system error: {e}")

        # Test 5: Logging System
        print("\n5. Testing Logging System...")
        try:
            # Test each logger
            logger.log_model_action("SYSTEM_TEST", "Testing logging functionality")
            logger.log_user_feedback("test", "spam", "spam", 0.9)
            logger.log_error("TEST_ERROR", "This is a test error")

            # Check if log files exist and have content
            log_files_exist = all(
                os.path.exists(f) and os.path.getsize(f) > 0
                for f in ['model_actions.log', 'user_feedback.log', 'error_reports.log']
            )

            if log_files_exist:
                print("   [PASS] All log files created and populated")
                test_results['logging_system'] = True
            else:
                print("   [FAIL] Log files missing or empty")
        except Exception as e:
            print(f"   [FAIL] Logging system error: {e}")

        # Test 6: Echo Logging System
        print("\n6. Testing Echo Logging System...")
        try:
            # Test echo logger functionality
            initial_state = echo_logger.echo_enabled
            initial_session = echo_logger.current_session

            # Enable echo logging
            if echo_logger.enable_echo_log():
                print("   [PASS] Echo logging enabled successfully")

                # Test logging an interaction
                echo_logger.log_interaction("TEST_INPUT", "test command")
                echo_logger.log_interaction("TEST_OUTPUT", "test response")

                # Check if echo log file exists and has content
                if os.path.exists('echo_interactions.log') and os.path.getsize('echo_interactions.log') > 0:
                    print("   [PASS] Echo interactions logged successfully")

                    # Disable echo logging
                    if echo_logger.disable_echo_log():
                        print("   [PASS] Echo logging disabled successfully")
                        test_results['echo_logging_system'] = True
                    else:
                        print("   [FAIL] Failed to disable echo logging")
                else:
                    print("   [FAIL] Echo log file not created or empty")
            else:
                print("   [FAIL] Failed to enable echo logging")

            # Restore initial state
            echo_logger.echo_enabled = initial_state
            echo_logger.current_session = initial_session
            echo_logger.save_session_state()

        except Exception as e:
            print(f"   [FAIL] Echo logging system error: {e}")

        # Test 7: Android Bridge
        print("\n7. Testing Android Bridge...")
        try:
            # Test Android bridge commands
            import subprocess
            result = subprocess.run([
                sys.executable, 'android_bridge.py', 'classify',
                'Test message for Android bridge'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and '"success": true' in result.stdout:
                print("   [PASS] Android bridge responding correctly")
                test_results['android_bridge'] = True
            else:
                print("   [FAIL] Android bridge test failed")
                print(f"   Output: {result.stdout[:100]}...")
        except Exception as e:
            print(f"   [FAIL] Android bridge error: {e}")

        # Test 8: Android Echo Bridge
        print("\n8. Testing Android Echo Bridge...")
        try:
            import subprocess

            # Test echo status
            result = subprocess.run([
                sys.executable, 'android_bridge.py', 'echo_status'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and '"enabled"' in result.stdout:
                print("   [PASS] Echo status command working")

                # Test enable echo
                result = subprocess.run([
                    sys.executable, 'android_bridge.py', 'enable_echo'
                ], capture_output=True, text=True, timeout=10)

                if result.returncode == 0 and '"Echo logging enabled"' in result.stdout:
                    print("   [PASS] Enable echo command working")

                    # Test disable echo
                    result = subprocess.run([
                        sys.executable, 'android_bridge.py', 'disable_echo'
                    ], capture_output=True, text=True, timeout=10)

                    if result.returncode == 0 and '"Echo logging disabled"' in result.stdout:
                        print("   [PASS] Disable echo command working")
                        test_results['android_echo_bridge'] = True
                    else:
                        print("   [FAIL] Disable echo command failed")
                else:
                    print("   [FAIL] Enable echo command failed")
            else:
                print("   [FAIL] Echo status command failed")
        except Exception as e:
            print(f"   [FAIL] Android echo bridge error: {e}")

        # Test 9: Performance Benchmark
        print("\n7. Testing Performance Benchmark...")
        try:
            # Run a quick benchmark
            metrics = self.benchmark.run_full_benchmark()
            if metrics and 'avg_time_per_message' in metrics:
                avg_time = metrics['avg_time_per_message']
                if avg_time < 1.0:  # Less than 1 second per message
                    print(f"   [PASS] Average time: {avg_time:.4f}s (acceptable)")
                    test_results['performance_test'] = True
                else:
                    print(f"   ⚠ Average time: {avg_time:.4f}s (slow)")
            else:
                print("   [FAIL] Performance test failed")
        except Exception as e:
            print(f"   [FAIL] Performance test error: {e}")

        # Test 9: Memory Usage
        print("\n9. Testing Memory Usage...")
        try:
            memory_stats = self.benchmark.benchmark_memory_usage()
            if memory_stats and 'baseline_memory' in memory_stats:
                baseline = memory_stats['baseline_memory']
                if baseline > 0 and baseline < 500:  # Reasonable memory usage in MB
                    print(f"   [PASS] Memory usage: {baseline:.1f} MB (acceptable)")
                    test_results['memory_test'] = True
                else:
                    print(f"   ⚠ Memory usage: {baseline:.1f} MB (high)")
            else:
                print("   [FAIL] Memory test failed")
        except Exception as e:
            print(f"   [FAIL] Memory test error: {e}")

        # Test 10: GUI Functionality
        print("\n10. Testing GUI Functionality...")
        try:
            gui_success = self.test_gui_functionality()
            test_results['gui_functionality'] = gui_success
        except Exception as e:
            print(f"   [FAIL] GUI functionality test error: {e}")
            test_results['gui_functionality'] = False

        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        passed_tests = sum(test_results.values())
        total_tests = len(test_results)

        for test_name, passed in test_results.items():
            status = "[PASS] PASS" if passed else "[FAIL] FAIL"
            print(f"  {test_name:<25} {status}")

        print("-" * 80)
        print(f"OVERALL RESULT: {passed_tests}/{total_tests} tests passed")

        if passed_tests >= total_tests * 0.8:  # 80% pass rate
            print("[PASS] SYSTEM TEST PASSED - Ready for deployment")
            return True
        else:
            print("[FAIL] SYSTEM TEST FAILED - Issues need to be resolved")
            return False

    def test_mobile_optimization(self):
        """Test mobile-specific optimizations."""
        print("\nTesting Mobile Optimizations...")

        # Test model size
        model_files = [f for f in os.listdir('.') if f.endswith('.pkl') and not f.endswith('_metadata.pkl')]
        if model_files:
            current_model = None
            if os.path.exists('current_model.txt'):
                with open('current_model.txt', 'r') as f:
                    current_model = f.read().strip()

            if current_model and os.path.exists(current_model):
                model_size = os.path.getsize(current_model) / (1024 * 1024)  # MB
                print(f"   Model size: {model_size:.2f} MB")
                if model_size < 10:  # Target < 10MB
                    print("   [PASS] Model size suitable for mobile devices")
                else:
                    print("   ⚠ Model size may be large for some mobile devices")

        # Test inference speed
        print("\nTesting inference speed for mobile...")
        mobile_simulation = self.benchmark.run_mobile_simulation()

        if mobile_simulation:
            print("   [PASS] Mobile simulation completed")
        else:
            print("   [FAIL] Mobile simulation failed")

    def run_integration_test(self):
        """Run integration test simulating real usage."""
        print("\nRunning Integration Test (Simulated Real Usage)...")

        # Simulate a user session
        test_messages = [
            "Hey, want to grab lunch tomorrow?",
            "WINNER! Click here to claim your prize!",
            "Meeting moved to 3 PM",
            "URGENT: Your PayPal account needs verification",
            "Thanks for the help!",
            "FREE iPhone giveaway! Enter now!"
        ]

        print("Simulating user interactions...")

        for i, msg in enumerate(test_messages, 1):
            print(f"\nMessage {i}: {msg}")

            # Classify
            result = self.classifier.classify_message(msg)
            if "error" in result:
                print(f"   Error: {result['error']}")
                continue

            classification = result['classification']
            confidence = result['confidence']

            print(f"   Classified as: {classification.upper()} (confidence: {confidence:.3f})")

            # Simulate feedback if low confidence
            if result.get('needs_feedback', False):
                # Auto-correct for test (assume model is correct for demo)
                self.feedback_system.log_feedback(msg, classification, classification, confidence)
                print("   [PASS] Feedback recorded (low confidence)")
            else:
                print("   [PASS] High confidence - no feedback needed")

            # Small delay to simulate real usage
            time.sleep(0.1)

        print("\nIntegration test completed successfully!")

    def test_gui_functionality(self):
        """Test GUI functionality by checking module import and method availability."""
        print("\nTesting GUI Functionality...")

        gui_tests_passed = 0
        total_gui_tests = 0

        try:
            # Test 1: Module import
            total_gui_tests += 1
            try:
                import sms_spam_detector_gui
                print("   [PASS] GUI module imports successfully")
                gui_tests_passed += 1
            except ImportError as e:
                print(f"   [FAIL] Could not import GUI module: {e}")
                return False

            # Test 2: Class availability
            total_gui_tests += 1
            if hasattr(sms_spam_detector_gui, 'SMSSpamDetectorGUI'):
                print("   [PASS] SMSSpamDetectorGUI class available")
                gui_tests_passed += 1
            else:
                print("   [FAIL] SMSSpamDetectorGUI class not found")
                return False

            # Test 3: Main function availability
            total_gui_tests += 1
            if hasattr(sms_spam_detector_gui, 'main'):
                print("   [PASS] Main function available")
                gui_tests_passed += 1
            else:
                print("   [FAIL] Main function not found")

            # Test 4: Check for key methods in the class
            total_gui_tests += 1
            gui_class = sms_spam_detector_gui.SMSSpamDetectorGUI
            key_methods = ['__init__', 'classify_message', 'load_current_model', 'update_status',
                          'refresh_model_list', 'update_stats_display', 'run_benchmark']

            methods_found = sum(1 for method in key_methods if hasattr(gui_class, method))
            if methods_found >= len(key_methods) * 0.8:  # At least 80% of methods present
                print(f"   [PASS] Key methods available ({methods_found}/{len(key_methods)})")
                gui_tests_passed += 1
            else:
                print(f"   [FAIL] Missing key methods ({methods_found}/{len(key_methods)})")

            # Test 5: Check for proper imports (syntax check)
            total_gui_tests += 1
            try:
                # Try to access imported modules to check they exist
                if hasattr(sms_spam_detector_gui, 'SMSClassifier') and \
                   hasattr(sms_spam_detector_gui, 'FeedbackSystem') and \
                   hasattr(sms_spam_detector_gui, 'logger'):
                    print("   [PASS] Required dependencies imported")
                    gui_tests_passed += 1
                else:
                    print("   [FAIL] Missing required dependencies")
            except Exception as e:
                print(f"   [FAIL] Import check error: {e}")

        except Exception as e:
            print(f"   [FAIL] GUI testing error: {e}")
            return False

        success_rate = gui_tests_passed / total_gui_tests if total_gui_tests > 0 else 0
        if success_rate >= 0.8:
            print(f"   [PASS] GUI tests: {gui_tests_passed}/{total_gui_tests} passed")
            return True
        else:
            print(f"   [FAIL] GUI tests: {gui_tests_passed}/{total_gui_tests} passed")
            return False

def main():
    """Main testing function."""
    tester = SystemTester()

    while True:
        print("\n" + "="*60)
        print("SYSTEM TESTING MENU")
        print("="*60)
        print("1. Run Full System Test")
        print("2. Test Mobile Optimizations")
        print("3. Run Integration Test")
        print("4. Quick Health Check")
        print("5. Back to main menu")

        choice = input("Enter your choice (1-5): ").strip()

        if choice == '1':
            success = tester.run_full_system_test()
            if success:
                print("\n[SUCCESS] All systems operational!")
            else:
                print("\n[WARN]  Some issues detected. Check logs for details.")

        elif choice == '2':
            tester.test_mobile_optimization()

        elif choice == '3':
            tester.run_integration_test()

        elif choice == '4':
            # Quick health check
            print("\nQuick Health Check:")
            if tester.classifier.model:
                print("[PASS] Model loaded")
            else:
                print("[FAIL] Model not loaded")

            if tester.classifier.vectorizer:
                print("[PASS] Vectorizer loaded")
            else:
                print("[FAIL] Vectorizer not loaded")

            log_files = ['model_actions.log', 'user_feedback.log', 'error_reports.log', 'echo_interactions.log']
            for log_file in log_files:
                if os.path.exists(log_file):
                    print(f"[PASS] {log_file} exists")
                else:
                    print(f"[FAIL] {log_file} missing")

            # Check echo logger status
            echo_status = "ENABLED" if echo_logger.echo_enabled else "DISABLED"
            print(f"[INFO] Echo logging: {echo_status} (Session: {echo_logger.current_session})")

        elif choice == '5':
            break
        else:
            print("Invalid choice. Please enter a number between 1-5.")

if __name__ == "__main__":
    main()