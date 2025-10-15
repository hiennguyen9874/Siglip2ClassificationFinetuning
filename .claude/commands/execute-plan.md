---
description: Execute a development plan with document-based task management
argument-hint: [plan-file-path]
---

# Execute Development Plan with Document-Based Task Management

You are about to execute a comprehensive development plan using document-based task tracking. This workflow ensures systematic task tracking and implementation throughout the entire development process using the built-in todo system.

## Critical Requirements

**MANDATORY**: Throughout the ENTIRE execution of this plan, you MUST maintain continuous usage of the todo system for task management. DO NOT drop or skip task tracking at any point. Every task from the plan must be tracked using TodoWrite from creation to completion.

## Step 1: Read and Parse the Plan

Read the plan file specified in: $ARGUMENTS

The plan file will contain:

-   A list of tasks to implement
-   References to existing codebase components and integration points
-   Context about where to look in the codebase for implementation

## Step 2: Task Setup and Planning

1. Read and understand the plan structure
    - Identify all tasks from the plan document
    - Understand task dependencies and sequencing
    - Note any special requirements or constraints

## Step 3: Create All Tasks in Todo System

For EACH task identified in the plan:

1. Create a corresponding task using `TodoWrite` with status "pending"
2. Include detailed descriptions from the plan
3. Maintain the task order/priority from the plan
4. Set up task dependencies where applicable

**IMPORTANT**: Create ALL tasks using TodoWrite upfront before starting implementation. This ensures complete visibility of the work scope and systematic tracking.

## Step 4: Codebase Analysis

Before implementation begins:

1. Analyze ALL integration points mentioned in the plan
2. Use Grep and Glob tools to:
    - Understand existing code patterns
    - Identify where changes need to be made
    - Find similar implementations for reference
3. Read all referenced files and components
4. Build a comprehensive understanding of the codebase context

## Step 5: Implementation Cycle

For EACH task in sequence:

### 5.1 Start Task

-   Move the current task to "in_progress" status using TodoWrite: `todo_write(merge=true, todos=[{id: "task_id", status: "in_progress"}])`
-   Use TodoWrite to track local subtasks if needed

### 5.2 Implement

-   Execute the implementation based on:
    -   The task requirements from the plan
    -   Your codebase analysis findings
    -   Best practices and existing patterns
-   Make all necessary code changes
-   Ensure code quality and consistency

### 5.3 Complete Task

-   Once implementation is complete, move task to "completed" status using TodoWrite: `todo_write(merge=true, todos=[{id: "task_id", status: "completed"}])`
-   Mark the task as done in the todo system for tracking purposes

### 5.4 Proceed to Next

-   Move to the next task in the list
-   Repeat steps 5.1-5.3

**CRITICAL**: Only ONE task should be in "in_progress" status at any time. Complete each task before starting the next.

## Step 6: Validation Phase

After ALL tasks are completed:

1. Create unit tests for the implemented functionality

    - Write focused unit tests for the main functionality
    - Test critical edge cases and error handling
    - Ensure tests follow the project's testing conventions
    - Use the project's existing test framework

2. Run the test suite

    - Execute the tests using the project's test runner
    - Verify all new tests pass
    - Check that existing tests still pass
    - Document any test failures and fix them

3. Perform integration validation
    - Check for integration issues between components
    - Ensure all acceptance criteria from the plan are met
    - Verify the implementation matches requirements
    - Test the complete workflow end-to-end

## Step 7: Final Validation and Documentation

After successful validation:

1. For each task that has corresponding unit test coverage:

    - Ensure the task remains in "completed" status in the todo system
    - Document test coverage in the task description if needed

2. For any tasks that need additional work:
    - Update task status to "pending" if rework is needed
    - Document what additional work is required

## Step 8: Final Report

Provide a summary including:

-   Total tasks created and completed
-   Any tasks remaining in review and why
-   Test coverage achieved
-   Key features implemented
-   Any issues encountered and how they were resolved

## Workflow Rules

1. **NEVER** skip todo system task management at any point
2. **ALWAYS** create all tasks using TodoWrite before starting implementation
3. **MAINTAIN** one task in "in_progress" status at a time
4. **VALIDATE** all work before marking tasks as "completed"
5. **TRACK** progress continuously through TodoWrite status updates
6. **ANALYZE** the codebase thoroughly before implementation
7. **TEST** everything before final completion

## Error Handling

If at any point TodoWrite operations fail:

1. Retry the operation
2. If persistent failures, document the issue but continue tracking locally
3. Never abandon task tracking - find workarounds if needed

Remember: The success of this execution depends on maintaining systematic task management through the todo system throughout the entire process. This ensures accountability, progress tracking, and quality delivery.
