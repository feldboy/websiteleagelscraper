# Branch Protection Rules Setup

## Required Branch Protection Rules

### Main Branch (Production)
```json
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "test",
      "deploy-pp"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 2,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true
  },
  "restrictions": {
    "users": [],
    "teams": ["senior-developers"],
    "apps": []
  }
}
```

### Pre-Production Branch (pp)
```json
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "test",
      "deploy-test"
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true
  }
}
```

### Test Branch
```json
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "test"
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1
  }
}
```

### Development Branch (dev)
```json
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "test"
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 0
  }
}
```

## Setup Instructions

1. Go to your GitHub repository settings
2. Navigate to "Branches" section
3. Add branch protection rule for each branch above
4. Configure the protection rules as specified
5. Save changes

## Workflow Protection

- **main**: Requires passing tests + pre-production deployment success
- **pp**: Requires passing tests + test environment deployment success
- **test**: Requires passing tests
- **dev**: Requires passing tests (minimal protection)