import argparse
import asyncio
import json
import os
import subprocess
from typing import Any

import litellm
from openhands.core.config import load_openhands_config
from openhands.core.logger import openhands_logger as logger
from openhands.resolver.issue_handler_factory import IssueHandlerFactory
from openhands.resolver.interfaces.issue import Issue
from openhands.resolver.utils import identify_token
from openhands.utils.async_utils import GENERAL_TIMEOUT, call_async_from_sync


class PRReviewer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.owner, self.repo = args.selected_repo.split('/')
        self.token = args.token or os.getenv('GITHUB_TOKEN')
        self.username = args.username or os.getenv('GIT_USERNAME') or 'openhands-agent'
        self.base_domain = args.base_domain or 'github.com'
        self.output_dir = args.output_dir
        self.issue_number = args.issue_number

        self.config = load_openhands_config()
        llm_config = self.config.get_llm_config()
        if args.llm_model:
            llm_config.model = args.llm_model
        if args.llm_api_key:
            from pydantic import SecretStr
            llm_config.api_key = SecretStr(args.llm_api_key)
        if args.llm_base_url:
            llm_config.base_url = args.llm_base_url
        self.config.set_llm_config(llm_config)

        factory = IssueHandlerFactory(
            owner=self.owner,
            repo=self.repo,
            token=self.token,
            username=self.username,
            platform=call_async_from_sync(identify_token, GENERAL_TIMEOUT, self.token, self.base_domain),
            base_domain=self.base_domain,
            issue_type='pr',
            llm_config=self.config.get_llm_config(),
        )
        self.issue_handler = factory.create()

    def get_diff(self, issue: Issue) -> str:
        repo_dir = os.path.join(self.output_dir, 'repo')
        if not os.path.exists(repo_dir):
            subprocess.check_call(
                ['git', 'clone', self.issue_handler.get_clone_url(), repo_dir]
            )

        # Fetch head branch
        subprocess.check_call(
            ['git', 'fetch', 'origin', issue.head_branch], cwd=repo_dir
        )

        # Get the diff against the base branch (usually main or master, but we should check issue.base_branch if available, though Issue model doesn't seem to have it populated by default in all handlers, let's assume main/master or try to find it)
        # Actually Issue model has base_branch, let's see if it's populated.
        # If not, we can try to determine default branch.

        base_branch = issue.base_branch or 'main' # Fallback
        subprocess.check_call(
            ['git', 'fetch', 'origin', base_branch], cwd=repo_dir
        )

        diff_cmd = ['git', 'diff', f'origin/{base_branch}...origin/{issue.head_branch}']
        diff = subprocess.check_output(diff_cmd, cwd=repo_dir).decode('utf-8')
        return diff

    async def review(self) -> None:
        issues = self.issue_handler.get_converted_issues([self.issue_number])
        if not issues:
            logger.error(f"Could not find PR #{self.issue_number}")
            return
        issue = issues[0]

        logger.info(f"Reviewing PR #{issue.number}: {issue.title}")

        try:
            diff = self.get_diff(issue)
        except Exception as e:
            logger.error(f"Failed to get diff: {e}")
            return

        if not diff.strip():
            logger.info("Empty diff, nothing to review.")
            return

        # Load prompt
        prompt_path = os.path.join(os.path.dirname(__file__), 'prompts/review/pr_review.jinja')
        with open(prompt_path, 'r') as f:
            prompt_template = f.read()

        # Simple jinja replacement (or use jinja2 if available)
        # Using simple replace for now to avoid extra deps if possible, but jinja2 is likely installed.
        from jinja2 import Template
        template = Template(prompt_template)
        prompt = template.render(issue=issue, diff=diff[:50000]) # Truncate diff if too large

        messages = [{"role": "user", "content": prompt}]

        response = await litellm.acompletion(
            model=self.config.get_llm_config().model,
            messages=messages,
            api_key=self.config.get_llm_config().api_key.get_secret_value() if self.config.get_llm_config().api_key else None,
            base_url=self.config.get_llm_config().base_url,
        )

        content = response.choices[0].message.content
        logger.info(f"LLM Response: {content}")

        # Parse JSON
        try:
            # content might be wrapped in ```json ... ```
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            review_data = json.loads(content)
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            # Fallback: post the raw content as a comment
            self.issue_handler.send_comment_msg(self.issue_number, f"**Automated Review**\n\n{content}")
            return

        # Post Summary
        summary = f"## Automated Code Review\n\n**Recommendation:** {review_data.get('recommendation', 'N/A')}\n\n{review_data.get('summary', '')}\n\n{review_data.get('general_feedback', '')}"
        self.issue_handler.send_comment_msg(self.issue_number, summary)

        # Post inline comments (if supported)
        # Currently IssueHandlerInterface doesn't support inline comments easily without GraphQL or specific API calls.
        # We can append them to the summary or try to implement it.
        # For now, let's append them to the summary or post as separate comments if they are specific.

        comments = review_data.get('comments', [])
        if comments:
            comments_text = "\n\n### Specific Feedback\n"
            for comment in comments:
                comments_text += f"- **{comment.get('file')}** (Line {comment.get('line')}): {comment.get('content')}\n"

            self.issue_handler.send_comment_msg(self.issue_number, comments_text)


def main():
    parser = argparse.ArgumentParser(description='Review a PR.')
    parser.add_argument('--selected-repo', required=True, help='owner/repo')
    parser.add_argument('--issue-number', type=int, required=True)
    parser.add_argument('--token', type=str)
    parser.add_argument('--username', type=str)
    parser.add_argument('--base-domain', type=str)
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--llm-model', type=str)
    parser.add_argument('--llm-api-key', type=str)
    parser.add_argument('--llm-base-url', type=str)

    args = parser.parse_args()

    reviewer = PRReviewer(args)
    asyncio.run(reviewer.review())


if __name__ == '__main__':
    main()
