"""
Test runner script with coverage reporting.
"""

import pytest
import sys
import coverage

def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    # Start coverage measurement
    cov = coverage.Coverage(
        source=['ml_pipeline'],
        omit=['*/tests/*', '*/venv/*']
    )
    cov.start()

    # Run tests
    result = pytest.main([
        '--verbose',
        '--color=yes',
        '--cov=ml_pipeline',
        '--cov-report=term-missing',
        'tests/'
    ])

    # Stop coverage measurement
    cov.stop()
    cov.save()

    # Generate coverage report
    print("\nCoverage Report:")
    cov.report()

    return result

if __name__ == '__main__':
    sys.exit(run_tests_with_coverage()) 