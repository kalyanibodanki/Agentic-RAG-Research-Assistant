
import time
import random


def call_gemini_with_retry(gemini_function, max_attempts=3):

    for attempt in range(max_attempts):

        try:
            return gemini_function()

        except Exception as error:

            error_message = str(error).upper()

            # Temporary server errors
            is_server_error = any(code in error_message for code in [
                "500", "502", "503", "504",
                "INTERNAL SERVER ERROR",
                "SERVICE UNAVAILABLE"
            ])

            # Temporary network and connection errors
            is_network_error = any(message in error_message for message in [
                "SERVER DISCONNECTED",
                "CONNECTION RESET",
                "CONNECTION ABORTED",
                "CONNECTION ERROR",
                "TIMED OUT",
                "TIMEOUT",
                "REMOTE PROTOCOL ERROR"
            ])

            # Rate-limit errors
            is_rate_limit = (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
            )

            # Do not retry exhausted daily quotas
            quota_exhausted = (
                is_rate_limit
                and any(message in error_message for message in [
                    "PER_DAY",
                    "DAILY",
                    "FREE_TIER",
                    "QUOTA EXCEEDED"
                ])
            )

            if quota_exhausted:
                print(
                    "Gemini quota appears exhausted. "
                    "Please check your API quota."
                )
                raise

            # Retry only temporary errors
            if not (
                is_server_error
                or is_network_error
                or is_rate_limit
            ):
                raise

            if attempt == max_attempts - 1:
                print("Gemini retry limit reached.")
                raise

            wait_time = (2 ** attempt) + random.uniform(0, 1)

            print(
                f"Temporary Gemini error: {error_message[:150]}"
            )
            print(
                f"Attempt {attempt + 1}/{max_attempts}. "
                f"Retrying in {wait_time:.1f} seconds..."
            )

            time.sleep(wait_time)
