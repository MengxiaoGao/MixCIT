from mixci._compat import *  # noqa: F401,F403

from mixci.benchmark import run_benchmark  # noqa: F401

def main():
        df = run_benchmark(
            sample_sizes=(500, 1000, 2000),
            mechanisms=('linear', 'nonlinear'),
            regimes=('CCC', 'DCC', 'CCD', 'DDD'),
            n_trials=100,
            alpha=0.05,
            n_jobs=-1,
            output_dir='results',
        )
        print("\nFinal summary:")


if __name__ == "__main__":
    main()
