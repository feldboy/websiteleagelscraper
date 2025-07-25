# Deployment Guide: Multi-Environment CI/CD with Railway

This guide covers the complete setup for deploying your Legal Research System across four environments using Railway and GitHub Actions.

## 🏗️ Environment Architecture

```
dev → test → pp → main
 ↓      ↓     ↓     ↓
Dev   Test   PP   Prod
```

### Environment Overview

| Environment | Branch | Purpose | Railway Service | Auto-Deploy |
|-------------|--------|---------|-----------------|-------------|
| **Development** | `dev` | Feature development | `legalresearch-dev` | ✅ |
| **Testing** | `test` | QA validation | `legalresearch-test` | ✅ |
| **Pre-Production** | `pp` | Final validation | `legalresearch-pp` | ✅ |
| **Production** | `main` | Live system | `legalresearch-prod` | ✅ |

## 🚀 Railway Setup

### 1. Create Railway Services

Create four separate services on Railway:

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Create services for each environment
railway service create legalresearch-dev
railway service create legalresearch-test
railway service create legalresearch-pp
railway service create legalresearch-prod
```

### 2. Configure Environment Variables

For each Railway service, configure these environment variables:

#### Development Service
```bash
railway service use legalresearch-dev
railway variables set -f .env.dev
```

#### Test Service
```bash
railway service use legalresearch-test
railway variables set -f .env.test
```

#### Pre-Production Service
```bash
railway service use legalresearch-pp
railway variables set -f .env.pp
```

#### Production Service
```bash
railway service use legalresearch-prod
railway variables set -f .env.prod
```

### 3. Database Setup

Each environment needs its own database:

```bash
# For each service, add PostgreSQL plugin
railway service use legalresearch-dev
railway add postgresql

railway service use legalresearch-test
railway add postgresql

railway service use legalresearch-pp
railway add postgresql

railway service use legalresearch-prod
railway add postgresql
```

### 4. Redis Setup

Add Redis for task queuing:

```bash
# For each service, add Redis plugin
railway service use legalresearch-dev
railway add redis

railway service use legalresearch-test
railway add redis

railway service use legalresearch-pp
railway add redis

railway service use legalresearch-prod
railway add redis
```

## ⚙️ GitHub Secrets Setup

Configure these secrets in your GitHub repository:

### Repository Secrets
```
RAILWAY_TOKEN=your-railway-api-token
RAILWAY_SERVICE_DEV=legalresearch-dev-service-id
RAILWAY_SERVICE_TEST=legalresearch-test-service-id
RAILWAY_SERVICE_PP=legalresearch-pp-service-id
RAILWAY_SERVICE_PROD=legalresearch-prod-service-id
```

### Environment-Specific Secrets

#### Development Environment
```
DATABASE_URL=postgresql-dev-url-from-railway
OPENAI_API_KEY=your-dev-openai-key
ANTHROPIC_API_KEY=your-dev-anthropic-key
TELEGRAM_BOT_TOKEN=your-dev-telegram-token
TELEGRAM_CHANNEL_ID=your-dev-channel-id
REDIS_URL=redis-dev-url-from-railway
```

#### Test Environment
```
DATABASE_URL=postgresql-test-url-from-railway
OPENAI_API_KEY=your-test-openai-key
ANTHROPIC_API_KEY=your-test-anthropic-key
TELEGRAM_BOT_TOKEN=your-test-telegram-token
TELEGRAM_CHANNEL_ID=your-test-channel-id
REDIS_URL=redis-test-url-from-railway
```

#### Pre-Production Environment
```
DATABASE_URL=postgresql-pp-url-from-railway
OPENAI_API_KEY=your-pp-openai-key
ANTHROPIC_API_KEY=your-pp-anthropic-key
TELEGRAM_BOT_TOKEN=your-pp-telegram-token
TELEGRAM_CHANNEL_ID=your-pp-channel-id
REDIS_URL=redis-pp-url-from-railway
```

#### Production Environment
```
DATABASE_URL=postgresql-prod-url-from-railway
OPENAI_API_KEY=your-prod-openai-key
ANTHROPIC_API_KEY=your-prod-anthropic-key
TELEGRAM_BOT_TOKEN=your-prod-telegram-token
TELEGRAM_CHANNEL_ID=your-prod-channel-id
REDIS_URL=redis-prod-url-from-railway
```

## 🌿 Branch Strategy & Workflow

### Branch Protection Rules

Set up branch protection in GitHub repository settings:

1. **main (Production)**:
   - Require pull request reviews (2 approvers)
   - Require status checks to pass
   - Require branches to be up to date
   - Include administrators in restrictions

2. **pp (Pre-Production)**:
   - Require pull request reviews (1 approver)
   - Require status checks to pass

3. **test**:
   - Require pull request reviews (1 approver)
   - Require status checks to pass

4. **dev**:
   - Require status checks to pass
   - No review required

### Development Workflow

1. **Feature Development**:
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/your-feature-name
   # Make changes
   git commit -m "feat: your feature description"
   git push origin feature/your-feature-name
   # Create PR to dev branch
   ```

2. **Promotion Flow**:
   ```
   dev → test → pp → main
   ```

3. **Automatic Deployments**:
   - Push to `dev` → Deploys to Railway dev environment
   - Push to `test` → Deploys to Railway test environment
   - Push to `pp` → Deploys to Railway pp environment
   - Push to `main` → Deploys to Railway production

## 🔄 CI/CD Pipeline

### Automated Workflows

1. **CI/CD Pipeline** (`.github/workflows/ci-cd.yml`):
   - Runs tests on all branches
   - Deploys to appropriate Railway environment
   - Includes linting, type checking, and coverage

2. **Branch Synchronization** (`.github/workflows/branch-sync.yml`):
   - Auto-creates PRs for promotion flow
   - Manual workflow dispatch for custom syncs

### Pipeline Stages

1. **Test Stage**:
   - Linting with flake8
   - Type checking with mypy  
   - Unit tests with pytest
   - Coverage reporting

2. **Deploy Stage**:
   - Environment-specific deployment
   - Railway service deployment
   - Health checks

## 🔧 Local Development

### Setup Local Environment

1. **Clone and setup**:
   ```bash
   git clone your-repo-url
   cd websiteleagelscraper
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment setup**:
   ```bash
   cp .env.example .env
   # Edit .env with your local configuration
   ```

3. **Database setup**:
   ```bash
   # Start local services with Docker
   docker-compose up -d postgres redis
   
   # Run migrations (when available)
   # alembic upgrade head
   ```

4. **Run application**:
   ```bash
   python main.py
   ```

### Testing

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test
pytest tests/test_agents/test_web_scraper.py -v
```

## 📊 Monitoring & Maintenance

### Environment Health Checks

Monitor your deployments:

1. **Railway Dashboard**: Check service health and logs
2. **GitHub Actions**: Monitor pipeline status
3. **Application Logs**: Check Railway service logs

### Database Migrations

When you need to update database schema:

```bash
# Create migration (when using Alembic)
alembic revision --autogenerate -m "description"

# Apply to development
railway run --service legalresearch-dev alembic upgrade head

# Apply to other environments after testing
```

### Rollback Strategy

If a deployment fails:

1. **Immediate**: Revert the problematic commit
2. **Railway**: Use Railway's rollback feature
3. **Database**: Run down migrations if needed

## 🔒 Security Considerations

1. **API Keys**: Use different keys for each environment
2. **Database**: Separate databases prevent data contamination  
3. **Telegram**: Use different bots/channels per environment
4. **Access Control**: Limit production access to senior developers

## 🚨 Troubleshooting

### Common Issues

1. **Deployment Fails**:
   - Check Railway service logs
   - Verify environment variables
   - Ensure dependencies are up to date

2. **Tests Fail**:
   - Check GitHub Actions logs
   - Verify test database connection
   - Update test data if schema changed

3. **Environment Variables**:
   - Ensure all required variables are set
   - Check Railway service configuration
   - Verify GitHub secrets are correct

### Support Commands

```bash
# Check Railway service status
railway status

# View service logs
railway logs

# Connect to database
railway connect postgresql

# Run one-off commands
railway run python manage.py shell
```

## 📝 Next Steps

1. Set up Railway services and configure environment variables
2. Configure GitHub repository secrets
3. Set up branch protection rules
4. Test the deployment pipeline
5. Train team on the new workflow
6. Set up monitoring and alerting

---

For additional help, refer to:
- [Railway Documentation](https://docs.railway.app/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Project README](./README.md)