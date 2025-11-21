import argparse
import asyncio
import os
from openhands.resolver.issue_resolver import IssueResolver
from openhands.core.logger import openhands_logger as logger

class PRReviewer(IssueResolver):
    async def review(self):
        # Run the resolver to process the PR
        output = await self.resolve_issue()

        if not output:
            logger.info("No output from resolve_issue (maybe already processed).")
            return

        # Post comment based on output
        # Note: IssueResolver might have already posted some comments if comment_success was True.
        # But we can add a summary comment here.

        summary = f"## OpenHands Agent Review\n\n"
        if output.success:
            summary += "✅ **Success**: The agent has successfully processed the PR.\n\n"
        else:
            summary += "❌ **Failure**: The agent encountered issues.\n\n"

        if output.result_explanation:
            summary += f"**Explanation:**\n{output.result_explanation}\n\n"

        if output.error:
            summary += f"**Error:**\n{output.error}\n\n"

        if output.git_patch:
            summary += f"**Git Patch Generated:**\n```diff\n{output.git_patch[:1000]}...\n```\n(Truncated)"

        self.issue_handler.send_comment_msg(self.issue_number, summary)


def main():
    parser = argparse.ArgumentParser(description='Review a PR using OpenHands agent.')
    parser.add_argument('--selected-repo', required=True, help='owner/repo')
    parser.add_argument('--issue-number', type=int, required=True)
    parser.add_argument('--token', type=str, help='GitHub token')
    parser.add_argument('--username', type=str, help='GitHub username')
    parser.add_argument('--base-domain', type=str, help='Base domain (e.g. github.com)')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--llm-model', type=str)
    parser.add_argument('--llm-api-key', type=str)
    parser.add_argument('--llm-base-url', type=str)
    parser.add_argument('--max-iterations', type=int, default=50)

    # Args expected by IssueResolver but not strictly needed for review if we set defaults
    parser.add_argument('--base-container-image', type=str)
    parser.add_argument('--runtime-container-image', type=str)
    parser.add_argument('--runtime', type=str, default='docker')
    parser.add_argument('--prompt-file', type=str)
    parser.add_argument('--repo-instruction-file', type=str)
    parser.add_argument('--is-experimental', action='store_true')
    parser.add_argument('--comment-id', type=int)

    args = parser.parse_args()

    # Set default issue type to 'pr'
    args.issue_type = 'pr'

    reviewer = PRReviewer(args)
    asyncio.run(reviewer.review())

if __name__ == '__main__':
    main()
