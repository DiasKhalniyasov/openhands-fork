# OpenHands PR Review Bot

This bot automatically reviews Pull Requests in the OpenHands repository using an LLM. It provides feedback on structure, style, bugs, and security.

## How it Works

The bot is integrated into the GitHub Actions workflow (`.github/workflows/openhands-resolver.yml`). It triggers on:
- New Pull Requests
- Updates to existing Pull Requests (synchronize)
- Reopened Pull Requests

It skips PRs created by `openhands-agent` or those with `[OpenHands]` in the title to avoid feedback loops.

## Automatic Usage

Simply open a Pull Request in the repository. The `Auto-Fix Tagged Issue with OpenHands / auto-review` job will start automatically.
Once completed, the bot will post a comment on your PR with its review summary and recommendations.

## Manual / Local Testing

You can run the review script locally to test changes or debug issues.

### Prerequisites

1.  **Python 3.12+**
2.  **Dependencies**: Install OpenHands dependencies.
    ```bash
    python venv -m openhands
    pip install -r requirements.txt
    # Or install the package in editable mode
    pip install -e .
    ```

### Environment Variables

You need to set the following environment variables:

-   `GITHUB_TOKEN`: A GitHub Personal Access Token (PAT) with `repo` scope.
-   `LLM_MODEL`: The LLM model to use (e.g., `anthropic/claude-3-5-sonnet-20240620`).
-   `LLM_API_KEY`: Your API key for the chosen LLM.
-   `LLM_BASE_URL`: (Optional) Base URL if using a custom provider.

### Running the Script

Use the following command to review a specific PR (replace placeholders with actual values):

```bash
python -m openhands.resolver.review_pr \
  --selected-repo OWNER/REPO \
  --issue-number PR_NUMBER \
  --token $GITHUB_TOKEN \
  --llm-model $LLM_MODEL \
  --llm-api-key $LLM_API_KEY
```

**Example:**

```bash
export GITHUB_TOKEN="your_github_token"
export LLM_API_KEY="your_anthropic_key"
export LLM_MODEL="anthropic/claude-3-5-sonnet-20240620"

python -m openhands.resolver.review_pr \
  --selected-repo OpenHands/OpenHands \
  --issue-number 123
```

### Output

The script will:
1.  Clone/Fetch the repository to a temporary `output/repo` directory.
2.  Get the diff for the specified PR.
3.  Send the diff to the LLM.
4.  Print the LLM's response to the console (logs).
5.  Attempt to post a comment on the PR (if the token has permissions).

## Customization

-   **Prompt**: You can modify the review instructions in `openhands/resolver/prompts/review/pr_review.jinja`.
-   **Logic**: The main logic is in `openhands/resolver/review_pr.py`.
