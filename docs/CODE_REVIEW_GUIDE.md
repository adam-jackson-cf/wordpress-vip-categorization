# Code Review Guide for Human Reviewers

## Overview

This project uses **automated AI code review** with GPT-5.1 Codex alongside human review. This guide explains how to work with the automated system and what human reviewers should focus on.

## Automated Review System

### How It Works

1. **Trigger**: When a PR is created or updated, GitHub Actions automatically runs an AI code review
2. **Model**: Uses GPT-5.1 Codex (latest OpenAI model for code review)
3. **Standards**: Reviews against `CODE_REVIEW.md` and `AGENTS.md` (when available)
4. **Output**: Single comprehensive review comment with all issues prioritized (Critical → Low)
5. **Quality Gates**: Separate job validates black, ruff, mypy, and 80% test coverage

### What the AI Reviews

- Quality gate compliance (black, ruff, mypy, pytest)
- Security vulnerabilities (hardcoded secrets, SQL injection, exposed data)
- Python code quality (imports, type hints, error handling)
- Pydantic patterns (Settings, models, validation)
- Database operations (SupabaseClient usage, parameterized queries)
- DSPy patterns (Signature design, optimization reproducibility)
- Testing patterns (AAA structure, mocking, coverage)
- Service layer patterns (dependency injection, logging)
- Performance issues (N+1 queries, missing async/await)

## Human Reviewer Responsibilities

### What Humans Should Focus On

AI reviews cover mechanical and pattern-based issues well, but humans should focus on:

#### 1. **Business Logic Correctness**
- Does the code actually solve the stated problem?
- Are edge cases handled appropriately?
- Is the algorithm correct for the use case?
- Does it match the requirements in the issue/ticket?

#### 2. **Architecture & Design Decisions**
- Does this fit the overall system architecture?
- Are there better design patterns for this use case?
- Will this scale with expected data volumes?
- Does it introduce unnecessary complexity?

#### 3. **Domain-Specific Knowledge**
- WordPress VIP API usage patterns
- Supabase-specific best practices
- OpenRouter/OpenAI API quirks
- DSPy optimization strategies for this specific use case

#### 4. **User Experience & Product Quality**
- CLI command usability
- Error message clarity for end users
- Logging appropriateness for debugging
- Export format correctness for downstream consumers

#### 5. **Context & Intent**
- Does the PR description match the actual changes?
- Is the approach reasonable given project constraints?
- Are there alternative solutions worth considering?
- Does this align with long-term project goals?

### Review Workflow

#### Step 1: Review AI Findings
1. Read the AI code review comment (posted automatically)
2. Verify critical issues are legitimate
3. Check if the developer addressed all critical/high-priority items

#### Step 2: Run Local Checks (Optional)
```bash
# Clone PR branch
gh pr checkout <pr-number>

# Run quality checks locally
make quality-check

# Run specific tests
pytest tests/unit/test_specific_file.py -v
```

#### Step 3: Human-Focused Review
Focus on the areas listed above (business logic, architecture, domain knowledge, UX, context)

#### Step 4: Provide Feedback
- **If AI found critical issues**: Ensure they're fixed before proceeding with human review
- **For human insights**: Add comments on design, architecture, business logic
- **For missed AI issues**: Report in PR comment and consider updating CODE_REVIEW.md

## Working with AI Review Feedback

### Priority Levels

AI reviews use this priority system:

- **🔴 CRITICAL**: Must fix before merge (security, quality gates, production-breaking)
- **🟠 HIGH**: Should fix before merge (missing tests, type safety, error handling)
- **🟡 MEDIUM**: Consider fixing (code style, minor performance)
- **🟢 LOW**: Optional improvements (naming, documentation)

### Responding to AI Reviews

#### If AI is Correct
1. Address all critical and high-priority issues
2. Push fixes to the PR
3. AI will re-review automatically
4. Mark resolved issues in PR comments for human reviewers

#### If AI is Wrong
1. Comment explaining why the AI is incorrect
2. Tag a human reviewer for confirmation
3. Consider whether CODE_REVIEW.md needs clarification to prevent future false positives

#### If Unsure
1. Ask clarifying questions in PR comments
2. Tag senior developers or subject matter experts
3. Consider adding a test case to validate the behavior

## Using GitHub Copilot `/review` Command

In addition to automated PR reviews, developers and reviewers can use the native `/review` command:

### In VS Code
1. Open a file or select code
2. Open GitHub Copilot Chat (Cmd+I / Ctrl+I)
3. Type `/review` or `/review against CODE_REVIEW.md`
4. Copilot will review based on CODE_REVIEW.md and AGENTS.md

### In GitHub Web UI
1. Open a PR
2. Comment: `@copilot review`
3. Copilot will review the PR

**Note**: Both methods read CODE_REVIEW.md and AGENTS.md automatically (as of Aug 2025)

## Improving the AI Review System

### When AI Misses Issues

If the AI consistently misses certain types of issues:

1. Document the pattern in `CODE_REVIEW.md`
2. Add specific examples to illustrate the issue
3. Consider adding to AGENTS.md if it's project-specific
4. Test by requesting AI review with `/review` command

### When AI Gives False Positives

If the AI incorrectly flags valid code:

1. Check if CODE_REVIEW.md is too strict or ambiguous
2. Consider adding clarifying examples
3. Update AGENTS.md with project-specific context
4. Document exceptions in CODE_REVIEW.md if pattern is intentionally different

### Updating Review Standards

To modify what the AI reviews:

1. Edit `CODE_REVIEW.md` - comprehensive standards
2. Edit `AGENTS.md` - project-specific patterns (when ready)
3. Edit `.github/copilot-review-prompt.md` - review instructions for GPT-5.1 Codex
4. Changes take effect on next PR

## Quality Gates vs AI Review

### Quality Gates (Automated)
- Black formatting
- Ruff linting
- MyPy type checking
- Pytest with 80% coverage

**These are non-negotiable and enforced by CI**

### AI Review (Guidance)
- Security patterns
- Code quality suggestions
- Best practice adherence
- Architecture recommendations

**These are strong recommendations but can be discussed**

### Human Review (Final Approval)
- Business logic correctness
- Design decisions
- Architecture fit
- UX/product quality

**Final decision on whether to merge**

## Common Scenarios

### Scenario 1: AI Flags Security Issue

```markdown
AI Comment:
🔴 CRITICAL - SQL injection vulnerability in search function
Location: src/services/matching.py:45
```

**Human reviewer should**:
1. ✅ Verify the issue is real
2. ✅ Ensure developer fixed it properly
3. ✅ Check if similar patterns exist elsewhere in the codebase
4. ✅ Approve only after fix is confirmed

### Scenario 2: AI Suggests Refactoring

```markdown
AI Comment:
🟡 MEDIUM - Consider extracting this 50-line function into smaller functions
Location: src/services/workflow.py:120-170
```

**Human reviewer should**:
1. ✅ Assess if refactoring improves readability
2. ✅ Consider if it's worth the risk of introducing bugs
3. ✅ Decide based on business context (e.g., is this code changing frequently?)
4. ✅ It's OK to defer if not critical

### Scenario 3: AI Misses Business Logic Bug

```markdown
Code:
def calculate_similarity(a, b):
    return sum(a) / sum(b)  # What if sum(b) is 0?
```

**Human reviewer should**:
1. ✅ Catch this edge case
2. ✅ Add comment requesting fix
3. ✅ Consider adding to CODE_REVIEW.md: "Check for division by zero"
4. ✅ Request test case for edge case

## Tips for Effective Review

### Do
- ✅ Read the PR description first
- ✅ Review AI findings before diving into code
- ✅ Focus on design and business logic
- ✅ Ask questions if something is unclear
- ✅ Suggest improvements with rationale
- ✅ Approve when quality bar is met

### Don't
- ❌ Duplicate what AI already flagged
- ❌ Focus only on style (AI handles this)
- ❌ Approve PRs with critical AI findings
- ❌ Dismiss AI feedback without investigation
- ❌ Nitpick on issues AI marked as LOW priority
- ❌ Merge without running quality gates

## Getting Help

### Questions About Review Standards
- Check `CODE_REVIEW.md` for comprehensive standards
- Check `AGENTS.md` for project-specific patterns (when available)
- Check `docs/DSPY_GEPA_BEST_PRACTICES.md` for DSPy patterns

### Questions About Automated Reviews
- Check `.github/workflows/code-review.yml` for workflow
- Check `.github/copilot-review-prompt.md` for AI instructions

### Questions About Project Architecture
- Check `README.md` for architecture overview
- Check `AGENTS.md` (when available) for component responsibilities

## Summary

**AI reviews handle**: Mechanical issues, pattern violations, quality gates, security basics
**Human reviews handle**: Business logic, architecture, domain expertise, UX, strategic decisions

**Best results**: AI + Human working together, each focusing on their strengths
