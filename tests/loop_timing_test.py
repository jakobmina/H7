import cl
import time
import numpy as np

RUN_FOR_SECONDS  = 30
TICKS_PER_SECOND = 1000
RUN_FOR_TICKS    = RUN_FOR_SECONDS * TICKS_PER_SECOND
PERCENTILES      = [0.001, 0.01, 0.1, 1, 99, 99.9, 99.99, 99.999]

def loop_timing_test():    
    """ Measure loop timing accuracy when using the CL Loop API. """
    
    loop_times_ns = np.empty(RUN_FOR_TICKS, dtype=int)
    spike_count   = 0

    with cl.open() as neurons:
        
        # Execute a loop that periodically returns spikes,
        # stims, and raw electrode data
        for tick in neurons.loop(
                        TICKS_PER_SECOND,
                        stop_after_ticks=RUN_FOR_TICKS,
                        ignore_jitter=True
                        ):
            
            # Capture the time that this loop iteration executes
            loop_times_ns[tick.iteration] = time.monotonic_ns()
            
            # Track the total number of spikes detected
            spike_count += len(tick.analysis.spikes)    
    
    # The loop is complete.
    # Now, report on loop interval times.
    
    intervals_us = np.diff(loop_times_ns) / 1000
    mean_us      = np.mean(intervals_us)
    min_us       = np.min(intervals_us)
    max_us       = np.max(intervals_us)
    percentiles  = \
        [(p, np.percentile(intervals_us, p)) for p in PERCENTILES]

    print(f"Ran WITH CL API for {RUN_FOR_SECONDS} seconds "
          f"at {TICKS_PER_SECOND} ticks per second "
          f"and {spike_count / RUN_FOR_SECONDS:.3f} spikes per second.")

    print(f"\nMean interval: {mean_us:.3f} µs, "
          f"(min: {min_us:.3f} µs, max: {max_us:.3f} µs)\n")

    print("Percentiles:\n")
    for percentile, value in percentiles:
        print(f"{percentile:>6.3f}: {value:>4.3f} µs")

if __name__ == "__main__":
    loop_timing_test()
