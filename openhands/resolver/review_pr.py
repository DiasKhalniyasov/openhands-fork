import argparse
import asyncio
import os
from argparse import Namespace

from openhands.core.logger import openhands_logger as logger
from openhands.resolver.interfaces.issue import Issue
from openhands.resolver.issue_resolver import IssueResolver


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
        1. Extracts the PR information
        2. Fetches the PR diff/changes
        3. Uses LLM to review the code
        4. Posts a comment with the review
        """
        logger.info(f'Starting PR review for #{self.issue_number}')

        # Extract PR information
        issue = self.extract_issue()

        # Fetch PR diff
        pr_diff = self._fetch_pr_diff()

        # Generate review using LLM
        review_summary = await self._generate_code_review(issue, pr_diff)

        # Post the review comment
        self._post_review_comment(review_summary)

        logger.info(f'Completed PR review for #{self.issue_number}')

    def _fetch_pr_diff(self) -> str:
        """Fetch the diff for the PR using GitHub API.

        Returns:
            The PR diff as a string
        """
        import requests

        logger.info(f'Fetching diff for PR #{self.issue_number}')

        # Get the PR diff using GitHub API
        url = f'{self.issue_handler._strategy.get_base_url()}/repos/{self.owner}/{self.repo}/pulls/{self.issue_number}'
        headers = self.issue_handler._strategy.get_headers()
        headers['Accept'] = 'application/vnd.github.v3.diff'

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        diff = response.text
        logger.info(f'Fetched diff: {len(diff)} characters')
        return diff

    async def _generate_code_review(self, issue: Issue, pr_diff: str) -> str:
        """Generate a code review using LLM.

        Args:
            issue: The Issue object containing PR information
            pr_diff: The PR diff string

        Returns:
            A formatted review with LLM-generated feedback
        """
        logger.info(f'Generating code review for PR #{issue.number}')

        # Build the prompt for the LLM
        prompt = self._build_review_prompt(issue, pr_diff)

        # Use the LLM to generate review
        llm_config = self.app_config.get_llm_config()
        from openhands.llm.llm import LLM

        llm = LLM(llm_config)

        try:
            response = llm.completion(
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are an expert code reviewer. Provide constructive, specific feedback on code changes.',
                    },
                    {'role': 'user', 'content': prompt},
                ]
            )

            review_content = response.choices[0].message.content
        except Exception as e:
            logger.error(f'Failed to generate LLM review: {e}')
            review_content = '*Unable to generate automated review. Please review manually.*'

        # Format the final review
        return self._format_review_output(issue, review_content, pr_diff)

    def _build_review_prompt(self, issue: Issue, pr_diff: str) -> str:
        """Build the prompt for LLM code review.

        Args:
            issue: The Issue object
            pr_diff: The PR diff

        Returns:
            The formatted prompt
        """
        # Truncate diff if too long (keep first 10000 chars)
        max_diff_length = 10000
        truncated_diff = pr_diff[:max_diff_length]
        if len(pr_diff) > max_diff_length:
            truncated_diff += '\n\n[... diff truncated for length ...]'

        prompt = f"""Please review the following pull request:

**Title:** {issue.title}
**PR Number:** #{issue.number}
**Repository:** {issue.owner}/{issue.repo}

**Description:**
{issue.body if issue.body else 'No description provided'}

**Code Changes:**
```diff
{truncated_diff}
```

Please provide a thorough code review covering:
1. **Code Quality**: Are there any code smells, anti-patterns, or best practice violations?
2. **Bugs & Issues**: Are there any potential bugs, edge cases, or logical errors?
3. **Security**: Are there any security vulnerabilities or concerns?
4. **Performance**: Are there any performance issues or optimization opportunities?
5. **Maintainability**: Is the code readable, well-structured, and maintainable?
6. **Testing**: Are there adequate tests? What test cases might be missing?

Format your response as a structured review with clear sections and actionable feedback.
"""
        return prompt

    def _format_review_output(
        self, issue: Issue, llm_review: str, pr_diff: str
    ) -> str:
        """Format the final review output.

        Args:
            issue: The Issue object
            llm_review: The LLM-generated review content
            pr_diff: The PR diff

        Returns:
            Formatted review string
        """
        output_parts = []

        # Header
        output_parts.append(f'## 🤖 AI Code Review: {issue.title}\n\n')
        output_parts.append(f'**PR #{issue.number}** in `{issue.owner}/{issue.repo}`\n\n')

        # Summary section
        if issue.body and issue.body.strip():
            output_parts.append('### 📋 PR Description\n')
            body_text = issue.body.strip()
            if len(body_text) > 500:
                output_parts.append(f'{body_text[:500]}...\n\n')
            else:
                output_parts.append(f'{body_text}\n\n')

        # Branch info
        if issue.head_branch and issue.base_branch:
            output_parts.append(
                f'**Branches:** `{issue.base_branch}` ← `{issue.head_branch}`\n\n'
            )

        # Changes summary
        diff_lines = pr_diff.split('\n')
        additions = sum(1 for line in diff_lines if line.startswith('+') and not line.startswith('+++'))
        deletions = sum(1 for line in diff_lines if line.startswith('-') and not line.startswith('---'))
        files_changed = len([line for line in diff_lines if line.startswith('diff --git')])

        output_parts.append(f'**Changes:** {files_changed} file(s) changed, ')
        output_parts.append(f'+{additions} additions, -{deletions} deletions\n\n')
        output_parts.append('---\n\n')

        # LLM Review
        output_parts.append('### 🔍 Review Feedback\n\n')
        output_parts.append(llm_review)
        output_parts.append('\n\n')

        # Footer
        output_parts.append('---\n')
        output_parts.append(
            '*🤖 Review generated by [ForteBank AI PR Reviewer] powered by AI*\n'
        )

        return ''.join(output_parts)

    async def _analyze_pr(self, issue: Issue) -> str:
        """Analyze the PR and generate a review.

        Args:
            issue: The Issue object containing PR information

        Returns:
            A formatted review summary
        """
        logger.info(f'Analyzing PR #{issue.number}: {issue.title}')

        # Build analysis of the PR
        analysis_parts = []

        # Add basic PR information
        analysis_parts.append(f'## PR Review: {issue.title}\n\n')
        analysis_parts.append(f'**PR #{issue.number}** in `{issue.owner}/{issue.repo}`\n\n')

        # Analyze PR description (always show, even if empty)
        analysis_parts.append('### Description\n')
        if issue.body and issue.body.strip():
            body_text = issue.body.strip()
            if len(body_text) > 1000:
                analysis_parts.append(f'{body_text[:1000]}...\n\n*[Truncated - full description available in PR]*\n\n')
            else:
                analysis_parts.append(f'{body_text}\n\n')
        else:
            analysis_parts.append('*No description provided*\n\n')

        # Add branches info (always show if available)
        if issue.head_branch and issue.base_branch:
            analysis_parts.append('### Branch Information\n')
            analysis_parts.append(f'- **Base branch**: `{issue.base_branch}`\n')
            analysis_parts.append(f'- **Head branch**: `{issue.head_branch}`\n\n')

        # Add closing issues if present
        if issue.closing_issues and len(issue.closing_issues) > 0:
            analysis_parts.append('### Related Issues\n')
            analysis_parts.append(f'This PR references/closes: {", ".join(issue.closing_issues)}\n\n')

        # Analyze review threads if present
        if issue.review_threads and len(issue.review_threads) > 0:
            analysis_parts.append(f'### Review Threads ({len(issue.review_threads)} found)\n')
            for idx, thread in enumerate(issue.review_threads[:5], 1):
                analysis_parts.append(f'{idx}. **Files**: {", ".join(thread.files)}\n')
                comment_text = thread.comment.strip()
                if len(comment_text) > 300:
                    analysis_parts.append(f'   **Comment**: {comment_text[:300]}...\n\n')
                else:
                    analysis_parts.append(f'   **Comment**: {comment_text}\n\n')
            if len(issue.review_threads) > 5:
                analysis_parts.append(f'*... and {len(issue.review_threads) - 5} more threads*\n\n')

        # Analyze review comments if present (but not in threads)
        if issue.review_comments and len(issue.review_comments) > 0:
            analysis_parts.append(f'### Review Comments ({len(issue.review_comments)} found)\n')
            for idx, comment in enumerate(issue.review_comments[:5], 1):
                comment_text = comment.strip()
                if len(comment_text) > 300:
                    analysis_parts.append(f'{idx}. {comment_text[:300]}...\n\n')
                else:
                    analysis_parts.append(f'{idx}. {comment_text}\n\n')
            if len(issue.review_comments) > 5:
                analysis_parts.append(f'*... and {len(issue.review_comments) - 5} more comments*\n\n')

        # Analyze thread comments if present
        if issue.thread_comments and len(issue.thread_comments) > 0:
            analysis_parts.append(f'### Discussion Thread ({len(issue.thread_comments)} comments)\n')
            for idx, comment in enumerate(issue.thread_comments[:3], 1):
                comment_text = comment.strip()
                if len(comment_text) > 300:
                    analysis_parts.append(f'{idx}. {comment_text[:300]}...\n\n')
                else:
                    analysis_parts.append(f'{idx}. {comment_text}\n\n')
            if len(issue.thread_comments) > 3:
                analysis_parts.append(f'*... and {len(issue.thread_comments) - 3} more comments*\n\n')

        # Add review status summary
        analysis_parts.append('### Review Status\n')
        total_feedback = sum([
            len(issue.review_threads or []),
            len(issue.review_comments or []),
            len(issue.thread_comments or [])
        ])
        if total_feedback > 0:
            analysis_parts.append(f'📝 This PR has **{total_feedback}** review feedback item(s) to address.\n\n')
        else:
            analysis_parts.append('✅ No review feedback found. This PR appears ready for review.\n\n')

        analysis_parts.append('---\n')
        analysis_parts.append('*🤖 Review generated by [ForteBank AI PR Reviewer]*\n')

        return ''.join(analysis_parts)

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
