import asyncio
import logging
import sys
from sentinel.agent.loop import run_clinical_query
from sentinel.agent.ollama_client import ResilientOllamaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

async def main():
    query = "Patient is 34F, two weeks of low mood, poor sleep, stopped eating. Mentions she sometimes thinks about not wanting to be here anymore. What does mhGAP say?"
    session_id = "test_cli_session"
    client = ResilientOllamaClient()
    
    print("--- Starting Clinical Query ---")
    async for step_data in run_clinical_query(query, session_id, client):
        if "token" in step_data:
            sys.stdout.write(step_data["token"])
            sys.stdout.flush()
        else:
            print(f"\n[Step Event] {step_data}")
    print("\n--- Query Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
