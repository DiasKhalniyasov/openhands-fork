import argparse
import asyncio
import os
from argparse import Namespace

from openhands.resolver.issue_resolver import IssueResolver
from openhands.resolver.resolver_output import ResolverOutput
from openhands.core.logger import openhands_logger as logger


class PRReviewer(IssueResolver):
    """Pull Request Reviewer that extends IssueResolver.

    This class provides functionality to review pull requests by running
    the OpenHands agent and posting a summary comment with the results.

    Inherits all initialization and configuration from IssueResolver, and adds
    PR-specific review functionality including summary generation and comment posting.
    """

    def __init__(self, args: Namespace) -> None:
        """Initialize the PRReviewer with the given parameters.

        Args:
            args: Namespace containing all required arguments inherited from IssueResolver,
                  including owner, repo, token, username, issue_number, etc.
        """
        super().__init__(args)
        logger.info(f'Initialized PRReviewer for PR #{self.issue_number}')

    async def review(self) -> None:
        """Execute the PR review process.

        This method orchestrates the full PR review workflow:
        1. Runs the resolver to process the PR
        2. Generates a summary of the results
        3. Posts a comment with the review summary
        """
        logger.info(f'Starting PR review for #{self.issue_number}')

        # Run the resolver to process the PR
        output = await self.resolve_issue()

        if not output:
            logger.info('No output from resolve_issue (maybe already processed).')
            return

        # Generate and post the review summary
        summary = self._generate_review_summary(output)
        self._post_review_comment(summary)

        logger.info(f'Completed PR review for #{self.issue_number}')

    def _generate_review_summary(self, output: ResolverOutput) -> str:
        """Generate a formatted summary of the PR review results.

        Args:
            output: The ResolverOutput containing the results of the PR processing

        Returns:
            A formatted markdown string containing the review summary
        """
        summary_parts = ['## OpenHands Agent Review\n\n']

        # Add success/failure status
        if output.success:
            summary_parts.append(
                '✅ **Success**: The agent has successfully processed the PR.\n\n'
            )
        else:
            summary_parts.append('❌ **Failure**: The agent encountered issues.\n\n')

        # Add explanation if available
        if output.result_explanation:
            summary_parts.append(f'**Explanation:**\n{output.result_explanation}\n\n')

        # Add error information if available
        if output.error:
            summary_parts.append(f'**Error:**\n{output.error}\n\n')

        # Add git patch preview if available
        if output.git_patch:
            patch_preview = self._format_git_patch_preview(output.git_patch)
            summary_parts.append(patch_preview)

        return ''.join(summary_parts)

    def _format_git_patch_preview(
        self, git_patch: str, max_length: int = 1000
    ) -> str:
        """Format a preview of the git patch for display in the comment.

        Args:
            git_patch: The full git patch string
            max_length: Maximum length of the patch preview (default: 1000)

        Returns:
            A formatted markdown code block with the patch preview
        """
        if len(git_patch) > max_length:
            preview = git_patch[:max_length]
            return f'**Git Patch Generated:**\n```diff\n{preview}...\n```\n(Truncated)'
        else:
            return f'**Git Patch Generated:**\n```diff\n{git_patch}\n```'

    def _post_review_comment(self, summary: str) -> None:
        """Post the review summary as a comment on the PR.

        Args:
            summary: The formatted summary to post
        """
        logger.info(f'Posting review comment for PR #{self.issue_number}')
        self.issue_handler.send_comment_msg(self.issue_number, summary)
        logger.info('Review comment posted successfully')


def main() -> None:
    """Main entry point for the PR reviewer CLI.

    Parses command-line arguments, initializes the PRReviewer, and executes
    the review process.
    """
    parser = argparse.ArgumentParser(
        description='Review a pull request using the OpenHands agent.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Review a PR with basic options
  python -m openhands.resolver.review_pr \\
    --selected-repo owner/repo \\
    --issue-number 123 \\
    --token $GITHUB_TOKEN \\
    --username myuser

  # Review a PR with custom LLM settings
  python -m openhands.resolver.review_pr \\
    --selected-repo owner/repo \\
    --issue-number 123 \\
    --token $GITHUB_TOKEN \\
    --username myuser \\
    --llm-model gpt-4 \\
    --llm-api-key $OPENAI_API_KEY
        """,
    )

    # Required arguments
    parser.add_argument(
        '--selected-repo',
        required=True,
        help='Repository in owner/repo format',
    )
    parser.add_argument(
        '--issue-number',
        type=int,
        required=True,
        help='PR number to review',
    )

    # Authentication arguments
    parser.add_argument(
        '--token',
        type=str,
        help='GitHub token (can also use GITHUB_TOKEN env var)',
    )
    parser.add_argument(
        '--username',
        type=str,
        help='GitHub username (can also use GIT_USERNAME env var)',
    )
    parser.add_argument(
        '--base-domain',
        type=str,
        help='Base domain for git server (e.g., github.com, gitlab.com)',
    )

    # Output and iteration settings
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for results (default: output)',
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=50,
        help='Maximum number of agent iterations (default: 50)',
    )

    # LLM configuration arguments
    parser.add_argument(
        '--llm-model',
        type=str,
        help='LLM model to use',
    )
    parser.add_argument(
        '--llm-api-key',
        type=str,
        help='API key for the LLM',
    )
    parser.add_argument(
        '--llm-base-url',
        type=str,
        help='Base URL for the LLM API',
    )

    # Container and runtime arguments
    parser.add_argument(
        '--base-container-image',
        type=str,
        help='Base container image to use',
    )
    parser.add_argument(
        '--runtime-container-image',
        type=str,
        help='Runtime container image to use',
    )
    parser.add_argument(
        '--runtime',
        type=str,
        default='docker',
        help='Runtime to use (default: docker)',
    )

    # Additional configuration arguments
    parser.add_argument(
        '--prompt-file',
        type=str,
        help='Path to custom prompt template file',
    )
    parser.add_argument(
        '--repo-instruction-file',
        type=str,
        help='Path to repository-specific instruction file',
    )
    parser.add_argument(
        '--is-experimental',
        action='store_true',
        help='Enable experimental features',
    )
    parser.add_argument(
        '--comment-id',
        type=int,
        help='Specific comment ID to focus on',
    )

    args = parser.parse_args()

    # Set issue type to PR for PR review
    args.issue_type = 'pr'

    logger.info('Initializing PR reviewer...')
    reviewer = PRReviewer(args)

    logger.info('Starting PR review process...')
    asyncio.run(reviewer.review())


if __name__ == '__main__':
    main()
