import argparse
import asyncio
import os
import re
import sys
from argparse import Namespace
from enum import Enum

from openhands.core.logger import openhands_logger as logger
from openhands.resolver.interfaces.issue import Issue
from openhands.resolver.issue_resolver import IssueResolver


class SecurityOutcome(Enum):
    """Possible outcomes of a security review."""
    SECURE = 'secure'
    VULNERABILITIES_FOUND = 'vulnerabilities_found'
    CRITICAL_VULNERABILITIES = 'critical_vulnerabilities'
    NEEDS_MANUAL_REVIEW = 'needs_manual_review'


# Default label configurations for security review
SECURITY_LABELS = {
    SecurityOutcome.SECURE: ['security-reviewed', 'security-approved'],
    SecurityOutcome.VULNERABILITIES_FOUND: ['security-reviewed', 'security-issues-found'],
    SecurityOutcome.CRITICAL_VULNERABILITIES: ['security-reviewed', 'security-critical', 'changes-requested'],
    SecurityOutcome.NEEDS_MANUAL_REVIEW: ['security-reviewed', 'needs-security-review'],
}

# Labels to remove when setting a new security status
SECURITY_MUTUALLY_EXCLUSIVE_LABELS = [
    'security-approved',
    'security-issues-found',
    'security-critical',
    'needs-security-review',
]


class SecurityPRReviewer(IssueResolver):
    """Security-focused Pull Request Reviewer that extends IssueResolver.

    This class provides functionality to review pull requests for security issues
    by running the OpenHands agent and posting a security-focused summary comment.

    Inherits all initialization and configuration from IssueResolver, and adds
    security-specific review functionality.
    """

    def __init__(self, args: Namespace) -> None:
        """Initialize the SecurityPRReviewer with the given parameters.

        Args:
            args: Namespace containing all required arguments inherited from IssueResolver,
                  including owner, repo, token, username, issue_number, etc.
        """
        super().__init__(args)
        logger.info(f'Initialized SecurityPRReviewer for PR #{self.issue_number}')

    async def review(self) -> None:
        """Execute the security PR review process.

        This method orchestrates the full security PR review workflow:
        1. Extracts the PR information
        2. Fetches the PR diff/changes
        3. Uses LLM to perform security review
        4. Determines the security outcome
        5. Posts a comment with the security findings
        6. Sets appropriate labels and MR status
        """
        logger.info(f'Starting security review for PR #{self.issue_number}')

        # Extract PR information
        issue = self.extract_issue()

        # Fetch PR diff
        pr_diff = self._fetch_pr_diff()

        # Generate security review using LLM
        review_summary, review_content = await self._generate_security_review(issue, pr_diff)

        # Determine security outcome from LLM response
        outcome = self._determine_security_outcome(review_content)
        logger.info(f'Security outcome for PR #{self.issue_number}: {outcome.value}')

        # Post the review comment
        self._post_review_comment(review_summary)

        # Set labels and MR status based on outcome
        self._apply_security_outcome(outcome)

        # Notify reviewers if this is an escalation (critical vulnerabilities)
        self._notify_reviewers_on_escalation(outcome)

        logger.info(f'Completed security review for PR #{self.issue_number}')

        # Exit with error code if critical vulnerabilities to block CI pipeline
        if outcome == SecurityOutcome.CRITICAL_VULNERABILITIES:
            logger.warning('Critical vulnerabilities found - exiting with error code to block pipeline')
            sys.exit(1)

    def _fetch_pr_diff(self) -> str:
        """Fetch the diff for the PR using GitHub API.

        Returns:
            The PR diff as a string
        """
        import requests

        logger.info(f'Fetching diff for PR #{self.issue_number}')

        # Get the PR diff using GitHub API
        # get_base_url() already returns: https://api.github.com/repos/{owner}/{repo}
        url = f'{self.issue_handler._strategy.get_base_url()}/pulls/{self.issue_number}'
        headers = self.issue_handler._strategy.get_headers()
        headers['Accept'] = 'application/vnd.github.v3.diff'

        logger.info(f'Fetching PR diff from: {url}')
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        diff = response.text
        logger.info(f'Fetched diff: {len(diff)} characters')
        return diff

    async def _generate_security_review(self, issue: Issue, pr_diff: str) -> tuple[str, str]:
        """Generate a security-focused code review using LLM.

        Args:
            issue: The Issue object containing PR information
            pr_diff: The PR diff string

        Returns:
            A tuple of (formatted security review output, raw LLM review content)
        """
        logger.info(f'Generating security review for PR #{issue.number}')

        # Build the prompt for the LLM
        prompt = self._build_security_prompt(issue, pr_diff)

        # Use the LLM to generate review
        llm_config = self.app_config.get_llm_config()
        from openhands.llm.llm import LLM

        llm = LLM(llm_config, service_id='security-reviewer')

        try:
            response = llm.completion(
                messages=[
                    {
                        'role': 'system',
                        'content': '''You are an expert security analyst and code auditor specializing in identifying security vulnerabilities in code changes.

Your primary focus is to identify:
- OWASP Top 10 vulnerabilities (Injection, Broken Authentication, Sensitive Data Exposure, XXE, Broken Access Control, Security Misconfiguration, XSS, Insecure Deserialization, Using Components with Known Vulnerabilities, Insufficient Logging)
- CWE (Common Weakness Enumeration) issues
- Hardcoded secrets, API keys, passwords, or tokens
- Insecure cryptographic implementations
- Authentication and authorization flaws
- Input validation and sanitization issues
- SQL injection, command injection, and path traversal vulnerabilities
- Cross-site scripting (XSS) and cross-site request forgery (CSRF) risks
- Insecure direct object references (IDOR)
- Race conditions and timing attacks
- Memory safety issues (buffer overflows, use-after-free)
- Insecure dependencies and supply chain risks

Be thorough, specific, and provide actionable remediation advice for each finding.
Rate each finding by severity: CRITICAL, HIGH, MEDIUM, LOW, or INFORMATIONAL.

At the end of your review, you MUST include a clear security verdict in the following format:
**SECURITY_VERDICT: SECURE** - if no security issues were found
**SECURITY_VERDICT: VULNERABILITIES_FOUND** - if LOW/MEDIUM/HIGH severity issues were found
**SECURITY_VERDICT: CRITICAL_VULNERABILITIES** - if CRITICAL severity issues were found
**SECURITY_VERDICT: NEEDS_MANUAL_REVIEW** - if the code is too complex or unclear for automated review''',
                    },
                    {'role': 'user', 'content': prompt},
                ]
            )

            review_content = response.choices[0].message.content
        except Exception as e:
            logger.error(f'Failed to generate security review: {e}')
            review_content = '*Unable to generate automated security review. Please review manually.*\n\n**SECURITY_VERDICT: NEEDS_MANUAL_REVIEW**'

        # Format the final review
        formatted_output = self._format_security_output(issue, review_content, pr_diff)
        return formatted_output, review_content

    def _build_security_prompt(self, issue: Issue, pr_diff: str) -> str:
        """Build the prompt for LLM security review.

        Args:
            issue: The Issue object
            pr_diff: The PR diff

        Returns:
            The formatted security-focused prompt
        """
        # Truncate diff if too long (keep first 15000 chars for security review)
        max_diff_length = 15000
        truncated_diff = pr_diff[:max_diff_length]
        if len(pr_diff) > max_diff_length:
            truncated_diff += '\n\n[... diff truncated for length ...]'

        prompt = f"""Perform a comprehensive security review of the following pull request:

**Title:** {issue.title}
**PR Number:** #{issue.number}
**Repository:** {issue.owner}/{issue.repo}

**Description:**
{issue.body if issue.body else 'No description provided'}

**Code Changes:**
```diff
{truncated_diff}
```

Please analyze these code changes for security vulnerabilities and provide a detailed security assessment covering:

1. **Injection Vulnerabilities**
   - SQL injection
   - Command injection
   - LDAP injection
   - XPath injection
   - Code injection

2. **Authentication & Authorization**
   - Authentication bypass
   - Broken access control
   - Privilege escalation
   - Session management issues
   - Insecure direct object references (IDOR)

3. **Data Security**
   - Hardcoded secrets/credentials
   - Sensitive data exposure
   - Insecure data storage
   - Missing encryption
   - PII/PHI handling issues

4. **Input Validation & Output Encoding**
   - Cross-site scripting (XSS)
   - Path traversal
   - Open redirects
   - Header injection
   - SSRF (Server-Side Request Forgery)

5. **Cryptography**
   - Weak algorithms
   - Insecure random number generation
   - Key management issues
   - Certificate validation

6. **Configuration & Dependencies**
   - Security misconfigurations
   - Insecure defaults
   - Known vulnerable dependencies
   - Debug/development code in production

7. **Other Security Concerns**
   - Race conditions
   - Resource exhaustion (DoS)
   - Information disclosure
   - Logging sensitive data
   - Error handling issues

For each finding, provide:
- **Severity**: CRITICAL / HIGH / MEDIUM / LOW / INFORMATIONAL
- **Location**: File and line number if identifiable
- **Description**: Clear explanation of the vulnerability
- **Impact**: Potential security impact if exploited
- **Remediation**: Specific fix recommendations

If no security issues are found, explicitly state that the code changes appear secure and explain why.

At the end of your review, you MUST include a security verdict line in exactly this format:
**SECURITY_VERDICT: SECURE** - if no security issues were found
**SECURITY_VERDICT: VULNERABILITIES_FOUND** - if LOW/MEDIUM/HIGH severity issues were found
**SECURITY_VERDICT: CRITICAL_VULNERABILITIES** - if CRITICAL severity issues were found that must be fixed
**SECURITY_VERDICT: NEEDS_MANUAL_REVIEW** - if the code requires manual security expert review
"""
        return prompt

    def _format_security_output(
        self, issue: Issue, llm_review: str, pr_diff: str
    ) -> str:
        """Format the final security review output.

        Args:
            issue: The Issue object
            llm_review: The LLM-generated security review content
            pr_diff: The PR diff

        Returns:
            Formatted security review string
        """
        output_parts = []

        # Header
        output_parts.append(f'## 🔒 Security Review: {issue.title}\n\n')
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

        # Security Review
        output_parts.append('### 🛡️ Security Findings\n\n')
        output_parts.append(llm_review)
        output_parts.append('\n\n')

        # Footer
        output_parts.append('---\n')
        output_parts.append(
            '*🔒 Security review generated by [ForteBank AI Security Reviewer] powered by AI*\n'
        )
        output_parts.append(
            '\n⚠️ *This automated security review is not a substitute for manual security assessment. '
            'Please ensure proper security testing is performed before deployment.*\n'
        )

        return ''.join(output_parts)

    def _post_review_comment(self, summary: str) -> None:
        """Post the security review summary as a comment on the PR.

        Args:
            summary: The formatted summary to post
        """
        logger.info(f'Posting security review comment for PR #{self.issue_number}')
        self.issue_handler.send_comment_msg(self.issue_number, summary)
        logger.info('Security review comment posted successfully')

    def _determine_security_outcome(self, review_content: str) -> SecurityOutcome:
        """Determine the security outcome from the LLM response.

        Args:
            review_content: The raw LLM security review content

        Returns:
            The determined SecurityOutcome
        """
        # Look for explicit security verdict in the response
        verdict_pattern = r'\*\*SECURITY_VERDICT:\s*(SECURE|VULNERABILITIES_FOUND|CRITICAL_VULNERABILITIES|NEEDS_MANUAL_REVIEW)\*\*'
        match = re.search(verdict_pattern, review_content, re.IGNORECASE)

        if match:
            verdict = match.group(1).upper()
            if verdict == 'SECURE':
                return SecurityOutcome.SECURE
            elif verdict == 'VULNERABILITIES_FOUND':
                return SecurityOutcome.VULNERABILITIES_FOUND
            elif verdict == 'CRITICAL_VULNERABILITIES':
                return SecurityOutcome.CRITICAL_VULNERABILITIES
            elif verdict == 'NEEDS_MANUAL_REVIEW':
                return SecurityOutcome.NEEDS_MANUAL_REVIEW

        # Fallback: analyze content for security indicators
        content_lower = review_content.lower()

        # Check for critical security issues
        critical_indicators = [
            'critical', 'sql injection', 'command injection', 'rce',
            'remote code execution', 'authentication bypass', 'hardcoded password',
            'hardcoded secret', 'hardcoded api key', 'privilege escalation',
        ]
        if any(indicator in content_lower for indicator in critical_indicators):
            # Check if it's actually found vs just mentioned in categories
            if any(phrase in content_lower for phrase in ['found', 'detected', 'identified', 'vulnerable']):
                return SecurityOutcome.CRITICAL_VULNERABILITIES

        # Check for non-critical vulnerabilities
        vulnerability_indicators = [
            'high', 'medium', 'vulnerability', 'xss', 'csrf', 'idor',
            'insecure', 'weakness', 'flaw', 'risk',
        ]
        if any(indicator in content_lower for indicator in vulnerability_indicators):
            if any(phrase in content_lower for phrase in ['found', 'detected', 'identified', 'issue']):
                return SecurityOutcome.VULNERABILITIES_FOUND

        # Check for secure indicators
        secure_indicators = [
            'no security issues', 'appears secure', 'no vulnerabilities',
            'secure', 'no issues found', 'looks good',
        ]
        if any(indicator in content_lower for indicator in secure_indicators):
            return SecurityOutcome.SECURE

        # Default to needs manual review if no clear verdict
        return SecurityOutcome.NEEDS_MANUAL_REVIEW

    def _apply_security_outcome(self, outcome: SecurityOutcome) -> None:
        """Apply labels and MR status based on the security outcome.

        Args:
            outcome: The determined security outcome
        """
        logger.info(f'Applying security outcome: {outcome.value}')

        # Get the strategy (handler) from issue_handler
        strategy = self.issue_handler._strategy

        # Remove mutually exclusive security labels first
        try:
            strategy.remove_mr_labels(self.issue_number, SECURITY_MUTUALLY_EXCLUSIVE_LABELS)
        except Exception as e:
            logger.warning(f'Failed to remove existing security labels: {e}')

        # Add new labels based on outcome
        labels_to_add = SECURITY_LABELS.get(outcome, [])
        if labels_to_add:
            try:
                strategy.add_mr_labels(self.issue_number, labels_to_add)
            except Exception as e:
                logger.error(f'Failed to add security labels: {e}')

        # Set MR status based on outcome
        try:
            if outcome == SecurityOutcome.SECURE:
                # Don't auto-approve for security - just add labels
                # Security approval should typically come from a human
                logger.info('Security review passed - labels added, no auto-approval for security')
            elif outcome == SecurityOutcome.CRITICAL_VULNERABILITIES:
                strategy.set_mr_status(
                    self.issue_number,
                    'request_changes',
                    'Critical security vulnerabilities found that must be addressed before merging.'
                )
            elif outcome == SecurityOutcome.VULNERABILITIES_FOUND:
                strategy.set_mr_status(
                    self.issue_number,
                    'request_changes',
                    'Security vulnerabilities found that should be reviewed and addressed.'
                )
            # For NEEDS_MANUAL_REVIEW, we don't change the approval status
        except Exception as e:
            logger.warning(f'Failed to set MR status: {e}')

    def _notify_reviewers_on_escalation(self, outcome: SecurityOutcome) -> None:
        """Notify assigned reviewers if security-critical escalation.

        Args:
            outcome: The security review outcome
        """
        if outcome != SecurityOutcome.CRITICAL_VULNERABILITIES:
            return

        try:
            strategy = self.issue_handler._strategy
            reviewers = strategy.get_mr_reviewers(self.issue_number)
            if reviewers:
                mentions = ' '.join([f'@{u}' for u in reviewers])
                msg = (
                    f'**🚨 SECURITY ALERT** {mentions}\n\n'
                    f'Critical security vulnerabilities have been found in this MR. '
                    f'This requires immediate security review before proceeding.\n\n'
                    f'The code review pipeline has been blocked until these issues are resolved.'
                )
                strategy.send_comment_msg(self.issue_number, msg)
                logger.info(f'Notified reviewers about security escalation: {reviewers}')
            else:
                logger.info('No reviewers assigned to notify about security escalation')
        except Exception as e:
            logger.warning(f'Failed to notify reviewers: {e}')


def main() -> None:
    """Main entry point for the security PR reviewer CLI.

    Parses command-line arguments, initializes the SecurityPRReviewer, and executes
    the security review process.
    """
    parser = argparse.ArgumentParser(
        description='Perform security review of a pull request using the OpenHands agent.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Security review a PR with basic options
  python -m openhands.resolver.security_pr \\
    --selected-repo owner/repo \\
    --issue-number 123 \\
    --token $GITHUB_TOKEN \\
    --username myuser

  # Security review with custom LLM settings
  python -m openhands.resolver.security_pr \\
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

    logger.info('Initializing security PR reviewer...')
    reviewer = SecurityPRReviewer(args)

    logger.info('Starting security review process...')
    asyncio.run(reviewer.review())


if __name__ == '__main__':
    main()
