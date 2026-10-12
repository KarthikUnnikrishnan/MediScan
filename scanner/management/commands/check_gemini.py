from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Check Gemini credentials, list available models, and test the configured model."

    def handle(self, *args, **options):
        api_key = getattr(settings, "GEMINI_API_KEY", "")
        if not api_key:
            self.stderr.write("FAIL: GEMINI_API_KEY is not set.")
            raise CommandError("Gemini API key is not configured.")

        try:
            from google import genai
            client = genai.Client(api_key=api_key)
        except Exception as exc:
            self.stderr.write(f"FAIL: Client initialization: {type(exc).__name__}: {exc}")
            raise CommandError("Could not initialize Gemini client.") from exc

        try:
            self.stdout.write("Models available to this key:")
            for model in client.models.list():
                self.stdout.write(f"  {getattr(model, 'name', str(model))}")
        except Exception as exc:
            self.stderr.write(f"FAIL: Model listing: {type(exc).__name__}: {exc}")
            raise CommandError("Could not list Gemini models.") from exc

        model = getattr(settings, "GEMINI_MODEL", "gemini-3.8-flash")
        try:
            client.models.generate_content(model=model, contents="Reply with OK.")
        except Exception as exc:
            self.stderr.write(f"FAIL: Test generation with {model}: {type(exc).__name__}: {exc}")
            raise CommandError(f"Gemini test failed for {model}.") from exc
        self.stdout.write(self.style.SUCCESS(f"PASS: Gemini test generation succeeded with {model}."))
        close = getattr(client, "close", None)
        if close:
            close()
