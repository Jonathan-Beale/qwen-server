#!/usr/bin/env python3
"""
Generate training data for AI-Assistant's extended tool set.

Focuses on:
- Planning & research tools (create_plan, think_step_by_step, compare_options, etc.)
- Web search with inline citations [1] format
- Memory & knowledge graph tools
- Reminder tools

Output format matches train_lora.py expectations.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta

random.seed(42)

# =============================================================================
# Tool XML Generation
# =============================================================================

def tool_call(tool_name: str, params: Dict[str, Any]) -> str:
    """Generate properly formatted tool call XML."""
    param_lines = []
    for key, value in params.items():
        if isinstance(value, list):
            items = "\n".join(f"<item>{item}</item>" for item in value)
            param_lines.append(f"<{key}>\n{items}\n</{key}>")
        elif isinstance(value, dict):
            nested = "\n".join(f"<{k}>{v}</{k}>" for k, v in value.items())
            param_lines.append(f"<{key}>\n{nested}\n</{key}>")
        else:
            param_lines.append(f"<{key}>{value}</{key}>")

    params_xml = "\n".join(param_lines)
    return f"""<tool_call>
<tool_name>{tool_name}</tool_name>
<parameters>
{params_xml}
</parameters>
</tool_call>"""


# =============================================================================
# Planning Scenarios - Extended
# =============================================================================

PLANNING_SCENARIOS = [
    {
        "user": "Help me plan a website redesign project",
        "plan": {
            "title": "Website Redesign Project",
            "goal": "Launch a modern, responsive website with improved UX",
            "steps": [
                "Audit current site performance and user feedback",
                "Create wireframes for new layout",
                "Design new visual style and components",
                "Implement responsive frontend",
                "Migrate content and test thoroughly",
                "Deploy to production and monitor"
            ]
        }
    },
    {
        "user": "I need to migrate our database from MySQL to PostgreSQL",
        "plan": {
            "title": "MySQL to PostgreSQL Migration",
            "goal": "Successfully migrate all data with zero downtime",
            "steps": [
                "Audit current MySQL schema and data types",
                "Set up PostgreSQL environment",
                "Create schema conversion scripts",
                "Develop data migration scripts",
                "Test migration on staging environment",
                "Plan cutover window and rollback strategy",
                "Execute production migration",
                "Verify data integrity and application functionality"
            ]
        }
    },
    {
        "user": "Plan out implementing user authentication for our app",
        "plan": {
            "title": "User Authentication Implementation",
            "goal": "Implement secure user authentication with JWT tokens",
            "steps": [
                "Design authentication flow and token strategy",
                "Create user database schema",
                "Implement registration endpoint",
                "Implement login endpoint with JWT generation",
                "Add middleware for token validation",
                "Implement password reset flow",
                "Add rate limiting and security headers",
                "Write tests for all auth endpoints"
            ]
        }
    },
    {
        "user": "Help me plan setting up a CI/CD pipeline",
        "plan": {
            "title": "CI/CD Pipeline Setup",
            "goal": "Automate testing and deployment for faster releases",
            "steps": [
                "Choose CI/CD platform (GitHub Actions, GitLab CI, etc.)",
                "Define build and test stages",
                "Set up automated testing on pull requests",
                "Configure staging deployment",
                "Add production deployment with approval gates",
                "Set up monitoring and rollback procedures"
            ]
        }
    },
    {
        "user": "I want to refactor our monolith into microservices",
        "plan": {
            "title": "Monolith to Microservices Refactoring",
            "goal": "Decompose monolithic application into scalable microservices",
            "steps": [
                "Identify service boundaries using domain analysis",
                "Map dependencies between components",
                "Extract first service (lowest coupling)",
                "Implement API gateway for routing",
                "Set up service discovery and communication",
                "Gradually extract remaining services",
                "Implement distributed tracing and logging",
                "Update deployment infrastructure"
            ]
        }
    },
    {
        "user": "Plan a data backup and disaster recovery strategy",
        "plan": {
            "title": "Backup and Disaster Recovery Strategy",
            "goal": "Ensure business continuity with reliable backups and recovery procedures",
            "steps": [
                "Inventory all critical data and systems",
                "Define RPO and RTO requirements",
                "Implement automated daily backups",
                "Set up off-site backup replication",
                "Document recovery procedures",
                "Test recovery process quarterly",
                "Train team on disaster recovery"
            ]
        }
    },
    {
        "user": "Help me plan building a REST API for our mobile app",
        "plan": {
            "title": "Mobile App REST API Development",
            "goal": "Build a scalable REST API to power iOS and Android apps",
            "steps": [
                "Define API endpoints and data models",
                "Set up project structure and dependencies",
                "Implement authentication and authorization",
                "Build core CRUD endpoints",
                "Add pagination, filtering, and sorting",
                "Implement rate limiting and caching",
                "Write API documentation (OpenAPI/Swagger)",
                "Deploy and set up monitoring"
            ]
        }
    },
    {
        "user": "Plan migrating our app to the cloud",
        "plan": {
            "title": "Cloud Migration",
            "goal": "Migrate on-premise application to AWS/GCP/Azure",
            "steps": [
                "Assess current infrastructure and dependencies",
                "Choose cloud provider and services",
                "Design cloud architecture",
                "Set up networking and security groups",
                "Migrate database to managed service",
                "Containerize application",
                "Deploy to cloud with load balancing",
                "Configure monitoring and alerts",
                "Decommission on-premise infrastructure"
            ]
        }
    },
    # New planning scenarios
    {
        "user": "Plan implementing a notification system",
        "plan": {
            "title": "Notification System Implementation",
            "goal": "Build multi-channel notification system (email, push, SMS)",
            "steps": [
                "Design notification data model and preferences",
                "Set up message queue (Redis/RabbitMQ)",
                "Implement email provider integration",
                "Add push notification support",
                "Integrate SMS gateway",
                "Build notification preferences UI",
                "Add notification history and tracking",
                "Implement rate limiting per user"
            ]
        }
    },
    {
        "user": "Help me plan a security audit for our application",
        "plan": {
            "title": "Application Security Audit",
            "goal": "Identify and remediate security vulnerabilities",
            "steps": [
                "Review authentication and authorization flows",
                "Audit input validation and sanitization",
                "Check for SQL injection vulnerabilities",
                "Test for XSS vulnerabilities",
                "Review API security (rate limiting, CORS)",
                "Audit dependency vulnerabilities",
                "Check secrets management",
                "Document findings and remediation plan"
            ]
        }
    },
    {
        "user": "Plan setting up monitoring and alerting",
        "plan": {
            "title": "Monitoring and Alerting Setup",
            "goal": "Implement comprehensive monitoring with actionable alerts",
            "steps": [
                "Define key metrics and SLOs",
                "Set up metrics collection (Prometheus/DataDog)",
                "Implement application logging (structured JSON)",
                "Configure log aggregation (ELK/Loki)",
                "Create dashboards for key services",
                "Define alert thresholds and escalation",
                "Set up PagerDuty/Opsgenie integration",
                "Document runbooks for common alerts"
            ]
        }
    },
    {
        "user": "I need to implement a payment system",
        "plan": {
            "title": "Payment System Implementation",
            "goal": "Integrate secure payment processing with Stripe",
            "steps": [
                "Set up Stripe account and API keys",
                "Implement checkout flow with Stripe Elements",
                "Handle payment webhooks",
                "Implement subscription management",
                "Add invoice generation",
                "Set up refund handling",
                "Implement PCI compliance measures",
                "Add payment analytics and reporting"
            ]
        }
    },
    {
        "user": "Plan implementing search functionality",
        "plan": {
            "title": "Search Implementation",
            "goal": "Add full-text search with Elasticsearch",
            "steps": [
                "Set up Elasticsearch cluster",
                "Design index mappings for searchable content",
                "Implement data sync from primary database",
                "Build search API with filters and facets",
                "Add autocomplete suggestions",
                "Implement search analytics",
                "Optimize relevance scoring",
                "Add search result highlighting"
            ]
        }
    },
    {
        "user": "Help me plan implementing real-time features",
        "plan": {
            "title": "Real-time Features Implementation",
            "goal": "Add real-time updates using WebSockets",
            "steps": [
                "Choose WebSocket library (Socket.io/ws)",
                "Set up WebSocket server",
                "Implement connection authentication",
                "Design event/message protocol",
                "Add presence detection (online/offline)",
                "Implement real-time notifications",
                "Add typing indicators and live updates",
                "Scale with Redis pub/sub"
            ]
        }
    },
    {
        "user": "Plan a performance optimization project",
        "plan": {
            "title": "Performance Optimization",
            "goal": "Improve application performance by 50%",
            "steps": [
                "Establish performance baselines",
                "Profile application to identify bottlenecks",
                "Optimize database queries and add indexes",
                "Implement caching strategy",
                "Optimize frontend bundle size",
                "Add CDN for static assets",
                "Implement lazy loading",
                "Set up performance monitoring"
            ]
        }
    },
    {
        "user": "I need to plan an API versioning strategy",
        "plan": {
            "title": "API Versioning Strategy",
            "goal": "Implement sustainable API versioning for backward compatibility",
            "steps": [
                "Evaluate versioning approaches (URL, header, query)",
                "Choose versioning strategy",
                "Design deprecation policy",
                "Implement version routing",
                "Create migration guides for clients",
                "Set up version-specific documentation",
                "Add deprecation warnings in responses",
                "Plan sunset timeline for old versions"
            ]
        }
    },
    {
        "user": "Plan implementing a file upload system",
        "plan": {
            "title": "File Upload System",
            "goal": "Build secure, scalable file upload with S3 storage",
            "steps": [
                "Set up S3 bucket with proper permissions",
                "Implement presigned URL generation",
                "Build upload API with multipart support",
                "Add file type validation",
                "Implement virus scanning",
                "Add image resizing for thumbnails",
                "Implement file metadata storage",
                "Add download tracking and analytics"
            ]
        }
    },
    {
        "user": "Help me plan a logging standardization effort",
        "plan": {
            "title": "Logging Standardization",
            "goal": "Implement consistent structured logging across all services",
            "steps": [
                "Define logging standards and levels",
                "Create shared logging library",
                "Implement structured JSON format",
                "Add correlation IDs for tracing",
                "Set up log aggregation pipeline",
                "Create log-based alerts",
                "Document logging best practices",
                "Migrate existing services to new standard"
            ]
        }
    },
]

# =============================================================================
# Think Step by Step Scenarios - Extended
# =============================================================================

THINK_STEP_BY_STEP_SCENARIOS = [
    {
        "user": "Should we use Redis or Memcached for caching?",
        "problem": "Choosing between Redis and Memcached for application caching",
        "steps": [
            "Redis supports more data structures (strings, hashes, lists, sets, sorted sets)",
            "Memcached is simpler and slightly faster for basic key-value operations",
            "Redis offers persistence options, Memcached is purely in-memory",
            "Redis supports pub/sub messaging which could be useful later",
            "Our application needs session storage and rate limiting",
            "Redis can handle both caching and these additional use cases"
        ],
        "conclusion": "Redis is the better choice as it covers caching plus our session and rate limiting needs",
        "confidence": "high"
    },
    {
        "user": "Is it worth rewriting our Python backend in Go?",
        "problem": "Evaluating whether to rewrite Python backend in Go for performance",
        "steps": [
            "Current Python backend handles 500 requests/sec, we need 2000/sec",
            "Profiling shows most time spent in I/O, not CPU-bound operations",
            "Go would improve CPU-bound operations but I/O is the bottleneck",
            "Adding async (asyncio) to Python could improve I/O throughput 3-4x",
            "Rewrite would take 6+ months and introduce bugs",
            "Team has Python expertise, limited Go experience"
        ],
        "conclusion": "Optimize Python with async first. Rewrite only if that's insufficient.",
        "confidence": "high"
    },
    {
        "user": "Should we build or buy an analytics dashboard?",
        "problem": "Deciding between building custom analytics or using a third-party tool",
        "steps": [
            "Building custom: Full control, exact features we need, no per-seat licensing",
            "Building custom: 3-4 months development time, ongoing maintenance burden",
            "Third-party (Metabase, Looker): Ready immediately, proven reliability",
            "Third-party: Monthly cost ~$500-2000, some customization limitations",
            "Our team is small (5 engineers) with limited frontend expertise",
            "Analytics is not our core competency or competitive advantage"
        ],
        "conclusion": "Use a third-party tool like Metabase. Focus engineering on core product.",
        "confidence": "medium",
        "caveats": ["May need custom solution if data privacy requirements change", "Costs scale with team size"]
    },
    {
        "user": "How should we handle the N+1 query problem in our API?",
        "problem": "Resolving N+1 database query performance issue",
        "steps": [
            "Currently fetching 100 users, then 100 separate queries for profiles",
            "This causes 101 database round trips instead of 2",
            "Option 1: Eager loading with JOIN - single query, may over-fetch",
            "Option 2: Batch loading (WHERE id IN [...]) - 2 queries, flexible",
            "Option 3: DataLoader pattern - batches within request, good for GraphQL",
            "Our API is REST, endpoints are well-defined",
            "Eager loading with JOIN is simplest for our use case"
        ],
        "conclusion": "Implement eager loading with LEFT JOIN for the users-profiles relationship",
        "confidence": "high"
    },
    {
        "user": "What's causing our memory leak in production?",
        "problem": "Diagnosing memory leak in Node.js production application",
        "steps": [
            "Memory grows from 200MB to 2GB over 24 hours",
            "Heap snapshots show growing array of event listeners",
            "Event listeners are attached in WebSocket connection handler",
            "Listeners are not removed when connections close",
            "Each connection adds 3 listeners, ~10K connections/day",
            "Listeners accumulate because removeListener is never called"
        ],
        "conclusion": "Add cleanup in WebSocket 'close' event to remove all listeners for that connection",
        "confidence": "high"
    },
    # New think step by step scenarios
    {
        "user": "Should we use a monorepo or polyrepo structure?",
        "problem": "Deciding between monorepo and polyrepo for our codebase",
        "steps": [
            "Team has 12 developers across 4 services",
            "Services share common libraries and types",
            "Currently polyrepo with duplicated code and version drift",
            "Monorepo would simplify shared code and atomic changes",
            "Monorepo requires better tooling (Nx, Turborepo, Bazel)",
            "Team has experience with Git but not monorepo tools"
        ],
        "conclusion": "Migrate to monorepo using Turborepo for simpler shared code management",
        "confidence": "medium",
        "caveats": ["Requires CI/CD updates", "Initial learning curve for tooling"]
    },
    {
        "user": "Is GraphQL right for our new API?",
        "problem": "Evaluating GraphQL vs REST for new API",
        "steps": [
            "Frontend team requests flexible data fetching",
            "Current REST API has 47 endpoints, many with over-fetching",
            "GraphQL reduces over-fetching with precise queries",
            "Mobile app would benefit from reduced payload sizes",
            "Team has no GraphQL experience",
            "Backend is Python/Django which has good GraphQL support (Graphene)"
        ],
        "conclusion": "Adopt GraphQL for new features, keep existing REST endpoints",
        "confidence": "medium",
        "caveats": ["Requires team training", "Need to implement proper authorization"]
    },
    {
        "user": "Should we adopt TypeScript for our JavaScript project?",
        "problem": "Evaluating TypeScript adoption for existing JavaScript codebase",
        "steps": [
            "Current codebase is 50K lines of JavaScript",
            "Bug rate is high, many runtime type errors in production",
            "TypeScript catches type errors at compile time",
            "Gradual adoption is possible with allowJs",
            "Team has 3 developers, 1 knows TypeScript",
            "IDE support significantly better with TypeScript"
        ],
        "conclusion": "Adopt TypeScript gradually, start with new files and strict mode for new code",
        "confidence": "high"
    },
    {
        "user": "Why is our Docker image so large?",
        "problem": "Diagnosing large Docker image size (2.5GB)",
        "steps": [
            "Base image is ubuntu:latest (77MB) - could use alpine (5MB)",
            "Node modules folder is 800MB - includes devDependencies",
            "Build artifacts from webpack are included (200MB)",
            "Multiple RUN commands create extra layers",
            "No .dockerignore file, copying unnecessary files",
            "Not using multi-stage builds"
        ],
        "conclusion": "Use multi-stage build with alpine, add .dockerignore, exclude devDependencies",
        "confidence": "high"
    },
    {
        "user": "Should we implement our own auth or use Auth0?",
        "problem": "Build vs buy decision for authentication",
        "steps": [
            "Auth0 costs $23/month for 1000 MAU, scales up from there",
            "Building custom auth takes 2-3 weeks minimum",
            "Custom auth requires ongoing security maintenance",
            "Auth0 provides MFA, social login, SSO out of the box",
            "We need SAML SSO for enterprise customers",
            "Team has limited security expertise"
        ],
        "conclusion": "Use Auth0 - the security risk and maintenance burden of custom auth isn't worth it",
        "confidence": "high"
    },
    {
        "user": "How should we handle database migrations in production?",
        "problem": "Designing safe database migration strategy",
        "steps": [
            "Current approach: run migrations during deployment downtime",
            "Downtime window is 5 minutes, some migrations take longer",
            "Option 1: Expand-contract pattern for backward compatible changes",
            "Option 2: Online schema change tools (pt-online-schema-change, gh-ost)",
            "Option 3: Blue-green deployment with synchronized migrations",
            "Most of our changes are adding columns, not modifying existing"
        ],
        "conclusion": "Use expand-contract pattern for additive changes, gh-ost for large table modifications",
        "confidence": "high"
    },
    {
        "user": "Should we use server-side rendering or client-side rendering?",
        "problem": "Choosing rendering strategy for new web application",
        "steps": [
            "Application is a B2B dashboard, not public-facing",
            "SEO is not a concern for internal dashboards",
            "Initial load time is important for user experience",
            "Dashboard has complex, interactive visualizations",
            "SSR adds server complexity and hosting costs",
            "CSR with good loading states provides adequate UX"
        ],
        "conclusion": "Use CSR (React SPA) with code splitting and loading states",
        "confidence": "high",
        "caveats": ["Reconsider if we add public marketing pages"]
    },
    {
        "user": "Why are our API response times increasing?",
        "problem": "Diagnosing gradual API latency increase over past month",
        "steps": [
            "P50 latency increased from 50ms to 200ms",
            "Database query times are stable",
            "Memory and CPU utilization are normal",
            "Checked for N+1 queries - none found",
            "External API calls show increased latency",
            "Third-party service (payment provider) degraded"
        ],
        "conclusion": "Payment provider API is the bottleneck - add caching and circuit breaker",
        "confidence": "high"
    },
]

# =============================================================================
# Compare Options Scenarios - Extended
# =============================================================================

COMPARE_OPTIONS_SCENARIOS = [
    {
        "user": "Compare React, Vue, and Svelte for our new frontend",
        "title": "Frontend Framework Comparison",
        "options": ["React", "Vue", "Svelte"],
        "criteria": ["Learning Curve", "Performance", "Ecosystem", "Hiring Pool"],
        "analysis": {
            "React": {
                "Learning Curve": "Moderate - JSX and hooks take time",
                "Performance": "Good with optimization",
                "Ecosystem": "Excellent - huge library selection",
                "Hiring Pool": "Excellent - most popular framework"
            },
            "Vue": {
                "Learning Curve": "Easy - familiar template syntax",
                "Performance": "Very good out of the box",
                "Ecosystem": "Good - growing rapidly",
                "Hiring Pool": "Good - popular in Asia/Europe"
            },
            "Svelte": {
                "Learning Curve": "Easy - minimal boilerplate",
                "Performance": "Excellent - compile-time optimization",
                "Ecosystem": "Limited - smaller community",
                "Hiring Pool": "Limited - fewer experienced devs"
            }
        },
        "recommendation": "React for enterprise projects with hiring needs, Svelte for performance-critical apps with stable team"
    },
    {
        "user": "Help me choose between AWS, GCP, and Azure",
        "title": "Cloud Provider Comparison",
        "options": ["AWS", "GCP", "Azure"],
        "criteria": ["Service Breadth", "Pricing", "ML/AI Tools", "Enterprise Support"],
        "analysis": {
            "AWS": {
                "Service Breadth": "Excellent - most comprehensive",
                "Pricing": "Complex but competitive",
                "ML/AI Tools": "Good - SageMaker, Bedrock",
                "Enterprise Support": "Excellent"
            },
            "GCP": {
                "Service Breadth": "Good - focused selection",
                "Pricing": "Simpler, sustained discounts",
                "ML/AI Tools": "Excellent - Vertex AI, TPUs",
                "Enterprise Support": "Good"
            },
            "Azure": {
                "Service Breadth": "Excellent - strong enterprise",
                "Pricing": "Competitive with EA discounts",
                "ML/AI Tools": "Good - Azure ML, OpenAI integration",
                "Enterprise Support": "Excellent - Microsoft ecosystem"
            }
        },
        "recommendation": "AWS for general workloads, GCP for ML-heavy projects, Azure if already using Microsoft stack"
    },
    {
        "user": "SQL vs NoSQL for our user data?",
        "title": "Database Type Comparison",
        "options": ["PostgreSQL", "MongoDB", "DynamoDB"],
        "criteria": ["Query Flexibility", "Scalability", "Schema Management", "Cost"],
        "analysis": {
            "PostgreSQL": {
                "Query Flexibility": "Excellent - complex JOINs, aggregations",
                "Scalability": "Good with read replicas",
                "Schema Management": "Strict - migrations required",
                "Cost": "Low - open source"
            },
            "MongoDB": {
                "Query Flexibility": "Good - flexible queries",
                "Scalability": "Excellent - built-in sharding",
                "Schema Management": "Flexible - schemaless",
                "Cost": "Medium - Atlas pricing"
            },
            "DynamoDB": {
                "Query Flexibility": "Limited - key-value focused",
                "Scalability": "Excellent - auto-scaling",
                "Schema Management": "Flexible - schemaless",
                "Cost": "Variable - pay per request"
            }
        },
        "recommendation": "PostgreSQL for complex relational data, DynamoDB for simple high-scale workloads"
    },
    # New comparison scenarios
    {
        "user": "Compare Docker Swarm vs Kubernetes vs Nomad",
        "title": "Container Orchestration Comparison",
        "options": ["Docker Swarm", "Kubernetes", "Nomad"],
        "criteria": ["Complexity", "Features", "Community", "Learning Curve"],
        "analysis": {
            "Docker Swarm": {
                "Complexity": "Low - simple setup",
                "Features": "Basic - core orchestration",
                "Community": "Declining - less active",
                "Learning Curve": "Easy - Docker native"
            },
            "Kubernetes": {
                "Complexity": "High - many components",
                "Features": "Comprehensive - industry standard",
                "Community": "Excellent - massive ecosystem",
                "Learning Curve": "Steep - many concepts"
            },
            "Nomad": {
                "Complexity": "Medium - simpler than K8s",
                "Features": "Good - flexible workloads",
                "Community": "Growing - HashiCorp backed",
                "Learning Curve": "Moderate - straightforward"
            }
        },
        "recommendation": "Kubernetes for large teams and complex needs, Nomad for simpler deployments"
    },
    {
        "user": "Compare Jest, Vitest, and Mocha for testing",
        "title": "JavaScript Testing Framework Comparison",
        "options": ["Jest", "Vitest", "Mocha"],
        "criteria": ["Speed", "Configuration", "Features", "Ecosystem"],
        "analysis": {
            "Jest": {
                "Speed": "Good - parallel execution",
                "Configuration": "Zero-config for most cases",
                "Features": "All-in-one: mocking, coverage, snapshots",
                "Ecosystem": "Excellent - widely adopted"
            },
            "Vitest": {
                "Speed": "Excellent - Vite-powered",
                "Configuration": "Minimal - Vite compatible",
                "Features": "Jest-compatible API",
                "Ecosystem": "Growing - newer option"
            },
            "Mocha": {
                "Speed": "Good - lightweight",
                "Configuration": "Flexible but manual setup",
                "Features": "Basic - needs additional libraries",
                "Ecosystem": "Mature - been around longer"
            }
        },
        "recommendation": "Vitest for Vite projects, Jest for React/general, Mocha for custom setups"
    },
    {
        "user": "Help me compare message queues: RabbitMQ vs Kafka vs SQS",
        "title": "Message Queue Comparison",
        "options": ["RabbitMQ", "Kafka", "SQS"],
        "criteria": ["Throughput", "Ordering", "Complexity", "Use Case"],
        "analysis": {
            "RabbitMQ": {
                "Throughput": "Good - 10K+ msg/sec",
                "Ordering": "Per-queue ordering",
                "Complexity": "Moderate - feature rich",
                "Use Case": "Task queues, RPC patterns"
            },
            "Kafka": {
                "Throughput": "Excellent - 100K+ msg/sec",
                "Ordering": "Per-partition ordering",
                "Complexity": "High - distributed system",
                "Use Case": "Event streaming, logs"
            },
            "SQS": {
                "Throughput": "Good - scales automatically",
                "Ordering": "FIFO queues available",
                "Complexity": "Low - managed service",
                "Use Case": "Simple queuing, AWS native"
            }
        },
        "recommendation": "SQS for simple AWS workloads, RabbitMQ for complex routing, Kafka for high-volume streaming"
    },
    {
        "user": "Compare FastAPI vs Django vs Flask",
        "title": "Python Web Framework Comparison",
        "options": ["FastAPI", "Django", "Flask"],
        "criteria": ["Performance", "Features", "Learning Curve", "Async Support"],
        "analysis": {
            "FastAPI": {
                "Performance": "Excellent - async native",
                "Features": "API-focused, auto docs",
                "Learning Curve": "Moderate - type hints",
                "Async Support": "Native - built-in"
            },
            "Django": {
                "Performance": "Good - optimized ORM",
                "Features": "Batteries included",
                "Learning Curve": "Moderate - conventions",
                "Async Support": "Partial - Django 4.1+"
            },
            "Flask": {
                "Performance": "Good - lightweight",
                "Features": "Minimal - choose your own",
                "Learning Curve": "Easy - simple core",
                "Async Support": "Via extensions"
            }
        },
        "recommendation": "FastAPI for APIs, Django for full-stack apps, Flask for microservices"
    },
    {
        "user": "Compare Terraform vs Pulumi vs CloudFormation",
        "title": "Infrastructure as Code Comparison",
        "options": ["Terraform", "Pulumi", "CloudFormation"],
        "criteria": ["Multi-cloud", "Language", "State Management", "Learning Curve"],
        "analysis": {
            "Terraform": {
                "Multi-cloud": "Excellent - provider ecosystem",
                "Language": "HCL - domain specific",
                "State Management": "Remote state, locking",
                "Learning Curve": "Moderate - HCL syntax"
            },
            "Pulumi": {
                "Multi-cloud": "Good - growing providers",
                "Language": "Real languages (Python, TS, Go)",
                "State Management": "Pulumi Cloud or self-hosted",
                "Learning Curve": "Easy if you know the language"
            },
            "CloudFormation": {
                "Multi-cloud": "AWS only",
                "Language": "YAML/JSON templates",
                "State Management": "AWS managed",
                "Learning Curve": "Moderate - verbose"
            }
        },
        "recommendation": "Terraform for multi-cloud, Pulumi for developers preferring real languages, CloudFormation for AWS-only"
    },
]

# =============================================================================
# Web Search Scenarios - Extended
# =============================================================================

WEB_SEARCH_SCENARIOS = [
    {
        "user": "What are the best practices for API rate limiting?",
        "query": "API rate limiting best practices 2024",
        "results": [
            {"title": "API Rate Limiting Best Practices", "url": "https://cloud.google.com/architecture/rate-limiting-strategies", "snippet": "Use token bucket algorithm for flexible rate limiting. Implement graduated responses."},
            {"title": "Rate Limiting Strategies", "url": "https://stripe.com/blog/rate-limiters", "snippet": "Stripe uses request rate limiters and concurrent request limiters. Always return retry-after headers."},
            {"title": "Building Resilient APIs", "url": "https://aws.amazon.com/blogs/architecture/rate-limiting", "snippet": "Different limits for different endpoints. Use sliding window counters."}
        ],
        "report": """Based on my research, here are the key best practices for API rate limiting:

**Algorithm Selection**
The token bucket algorithm is widely recommended for its flexibility [1]. It allows for burst traffic while maintaining overall rate limits.

**Implementation Strategies**
- Use graduated responses with warning headers before hard limits [1]
- Combine request rate limiters with concurrent request limiters [2]
- Always return `Retry-After` headers to help clients back off appropriately [2]

**Advanced Considerations**
- Set different limits for different endpoints based on resource cost [3]
- Use sliding window counters for more accurate rate tracking [3]

Sources:
[1] https://cloud.google.com/architecture/rate-limiting-strategies
[2] https://stripe.com/blog/rate-limiters
[3] https://aws.amazon.com/blogs/architecture/rate-limiting"""
    },
    {
        "user": "Research the latest trends in serverless computing",
        "query": "serverless computing trends 2024",
        "results": [
            {"title": "State of Serverless 2024", "url": "https://www.datadoghq.com/state-of-serverless/", "snippet": "Serverless adoption grew 40% YoY. AWS Lambda dominates with 70% market share."},
            {"title": "Serverless Architecture Patterns", "url": "https://martinfowler.com/articles/serverless.html", "snippet": "Event-driven architectures are the sweet spot. Cold starts remain a challenge."},
            {"title": "The Future of Serverless", "url": "https://techcrunch.com/serverless-future/", "snippet": "Edge computing integration is the next frontier. WebAssembly enabling new use cases."}
        ],
        "report": """Here's what I found about serverless computing trends:

**Market Growth**
Serverless adoption has grown 40% year-over-year, with AWS Lambda maintaining dominant market share at approximately 70% [1].

**Architecture Patterns**
Event-driven architectures remain the sweet spot for serverless applications [2]. However, cold starts continue to be a challenge for latency-sensitive applications [2].

**Emerging Trends**
- Edge computing integration is becoming a major focus [3]
- WebAssembly (WASM) is enabling new serverless use cases [3]

Sources:
[1] https://www.datadoghq.com/state-of-serverless/
[2] https://martinfowler.com/articles/serverless.html
[3] https://techcrunch.com/serverless-future/"""
    },
    {
        "user": "What's the current state of WebAssembly adoption?",
        "query": "WebAssembly WASM adoption 2024",
        "results": [
            {"title": "WebAssembly Survey 2024", "url": "https://blog.aspect.dev/webassembly-survey-2024", "snippet": "78% of developers have used or plan to use WASM."},
            {"title": "WASM Beyond the Browser", "url": "https://bytecodealliance.org/articles/wasm-server-side", "snippet": "WASI enabling server-side WASM. Docker supports WASM containers."},
            {"title": "WebAssembly Languages", "url": "https://webassembly.org/languages/", "snippet": "Rust leads at 45%, C++ 25%, AssemblyScript 15%."}
        ],
        "report": """Here's the current state of WebAssembly adoption:

**Developer Adoption**
78% of developers have either used WebAssembly or plan to use it [1].

**Beyond the Browser**
WASI (WebAssembly System Interface) is enabling significant server-side adoption [2]. Docker now supports WASM containers natively [2].

**Language Ecosystem**
- Rust: 45% of WASM projects [3]
- C++: 25% [3]
- AssemblyScript: 15% [3]

Sources:
[1] https://blog.aspect.dev/webassembly-survey-2024
[2] https://bytecodealliance.org/articles/wasm-server-side
[3] https://webassembly.org/languages/"""
    },
    {
        "user": "How do I implement OAuth 2.0 properly?",
        "query": "OAuth 2.0 implementation best practices security",
        "results": [
            {"title": "OAuth 2.0 Security Best Practices", "url": "https://oauth.net/2/security-best-practices/", "snippet": "Always use PKCE for public clients. Authorization code flow recommended."},
            {"title": "Implementing OAuth 2.0", "url": "https://auth0.com/docs/authentication-flow", "snippet": "Use short-lived access tokens. Implement refresh token rotation."},
            {"title": "Common OAuth Vulnerabilities", "url": "https://portswigger.net/web-security/oauth", "snippet": "CSRF attacks via state parameter. Token leakage through referrer."}
        ],
        "report": """Here's how to implement OAuth 2.0 securely:

**Flow Selection**
- Always use Authorization Code flow with PKCE for public clients [1]
- Implicit flow is deprecated due to security concerns [1]

**Token Management**
- Use short-lived access tokens (5-15 minutes) [2]
- Implement refresh token rotation [2]
- Never store tokens in localStorage [2]

**Security Considerations**
- Always use the `state` parameter to prevent CSRF [3]
- Be aware of token leakage through referrer headers [3]

Sources:
[1] https://oauth.net/2/security-best-practices/
[2] https://auth0.com/docs/authentication-flow
[3] https://portswigger.net/web-security/oauth"""
    },
    {
        "user": "What are the best Python web frameworks in 2024?",
        "query": "best Python web frameworks 2024 comparison",
        "results": [
            {"title": "Python Web Frameworks Comparison", "url": "https://testdriven.io/blog/python-web-frameworks/", "snippet": "FastAPI leads for APIs. Django for full-stack. Flask for simplicity."},
            {"title": "FastAPI vs Django vs Flask", "url": "https://realpython.com/python-web-frameworks/", "snippet": "FastAPI: 3x faster than Flask. Django: batteries included. Flask: minimal."},
            {"title": "Python Framework Benchmarks", "url": "https://web-frameworks-benchmark.netlify.app/", "snippet": "FastAPI 15K req/sec. Django 2.5K. Flask 4K."}
        ],
        "report": """Here's my research on Python web frameworks:

**FastAPI** - Best for APIs
- 15,000 requests/second [3]
- Automatic OpenAPI documentation [1]
- 3x faster than Flask [2]

**Django** - Best for Full-Stack
- Batteries included: ORM, admin, auth [2]
- ~2,500 requests/second [3]

**Flask** - Best for Simplicity
- Minimal and flexible [2]
- ~4,000 requests/second [3]

**Recommendation**: FastAPI for APIs, Django for full-stack, Flask for microservices.

Sources:
[1] https://testdriven.io/blog/python-web-frameworks/
[2] https://realpython.com/python-web-frameworks/
[3] https://web-frameworks-benchmark.netlify.app/"""
    },
    {
        "user": "Research Kubernetes best practices for production",
        "query": "Kubernetes production best practices 2024",
        "results": [
            {"title": "K8s Production Checklist", "url": "https://kubernetes.io/docs/production/", "snippet": "Use namespaces. Set resource limits. Use network policies."},
            {"title": "K8s Security Best Practices", "url": "https://sysdig.com/kubernetes-security/", "snippet": "Run as non-root. Use RBAC. Scan images."},
            {"title": "K8s Reliability Patterns", "url": "https://learnk8s.io/production-best-practices", "snippet": "Pod anti-affinity. Liveness probes. HPA."}
        ],
        "report": """Here are Kubernetes production best practices:

**Resource Management**
- Always set resource requests and limits [1]
- Use namespaces for isolation [1]

**High Availability**
- Use pod anti-affinity to spread replicas [3]
- Configure PodDisruptionBudgets [1]
- Implement liveness and readiness probes [3]

**Security**
- Run containers as non-root [2]
- Implement RBAC with least privilege [2]
- Scan container images for vulnerabilities [2]

Sources:
[1] https://kubernetes.io/docs/production/
[2] https://sysdig.com/kubernetes-security/
[3] https://learnk8s.io/production-best-practices"""
    },
    # New web search scenarios
    {
        "user": "What's new in Python 3.12?",
        "query": "Python 3.12 new features",
        "results": [
            {"title": "Python 3.12 Release Notes", "url": "https://docs.python.org/3.12/whatsnew/3.12.html", "snippet": "Improved error messages. Per-interpreter GIL. Type parameter syntax."},
            {"title": "Python 3.12 Performance", "url": "https://speed.python.org/", "snippet": "5% faster on average. Better memory usage. Faster startup."},
            {"title": "Python 3.12 for Developers", "url": "https://realpython.com/python312-new-features/", "snippet": "F-string improvements. Better debugging. New typing features."}
        ],
        "report": """Here's what's new in Python 3.12:

**Performance Improvements**
- 5% faster on average than 3.11 [2]
- Better memory usage and faster startup [2]

**New Features**
- Improved error messages with more context [1]
- Per-interpreter GIL for better parallelism [1]
- New type parameter syntax for generics [1]
- F-string improvements allowing nested expressions [3]

**Developer Experience**
- Better debugging capabilities [3]
- New typing features for cleaner code [3]

Sources:
[1] https://docs.python.org/3.12/whatsnew/3.12.html
[2] https://speed.python.org/
[3] https://realpython.com/python312-new-features/"""
    },
    {
        "user": "Research best practices for microservices communication",
        "query": "microservices communication patterns best practices",
        "results": [
            {"title": "Microservices Communication", "url": "https://microservices.io/patterns/communication-style/", "snippet": "Sync: REST, gRPC. Async: messaging, events. Choose based on use case."},
            {"title": "gRPC vs REST", "url": "https://cloud.google.com/blog/products/api-management/grpc-vs-rest", "snippet": "gRPC 10x faster for internal services. REST better for public APIs."},
            {"title": "Event-Driven Architecture", "url": "https://martinfowler.com/articles/201701-event-driven.html", "snippet": "Event sourcing for audit. CQRS for read/write separation."}
        ],
        "report": """Here are best practices for microservices communication:

**Synchronous Communication**
- Use gRPC for internal service-to-service calls (10x faster than REST) [2]
- Use REST for public-facing APIs [2]

**Asynchronous Communication**
- Use message queues for decoupled, reliable delivery [1]
- Consider event-driven architecture for loose coupling [1]

**Advanced Patterns**
- Event sourcing for complete audit trails [3]
- CQRS for separating read and write models [3]

Sources:
[1] https://microservices.io/patterns/communication-style/
[2] https://cloud.google.com/blog/products/api-management/grpc-vs-rest
[3] https://martinfowler.com/articles/201701-event-driven.html"""
    },
    {
        "user": "How do I optimize PostgreSQL performance?",
        "query": "PostgreSQL performance tuning optimization",
        "results": [
            {"title": "PostgreSQL Performance Tuning", "url": "https://wiki.postgresql.org/wiki/Performance_Optimization", "snippet": "Tune shared_buffers, work_mem. Use EXPLAIN ANALYZE. Create proper indexes."},
            {"title": "PostgreSQL Indexing", "url": "https://use-the-index-luke.com/postgresql", "snippet": "B-tree for equality. GIN for full-text. Partial indexes for filtered queries."},
            {"title": "PostgreSQL Query Optimization", "url": "https://www.cybertec-postgresql.com/en/postgresql-query-optimization/", "snippet": "Avoid SELECT *. Use connection pooling. Batch operations."}
        ],
        "report": """Here's how to optimize PostgreSQL performance:

**Configuration Tuning**
- Set `shared_buffers` to 25% of RAM [1]
- Tune `work_mem` based on query complexity [1]
- Use `EXPLAIN ANALYZE` to understand query plans [1]

**Indexing Strategies**
- B-tree indexes for equality and range queries [2]
- GIN indexes for full-text search [2]
- Partial indexes for commonly filtered queries [2]

**Query Optimization**
- Avoid `SELECT *` - specify needed columns [3]
- Use connection pooling (PgBouncer) [3]
- Batch INSERT/UPDATE operations [3]

Sources:
[1] https://wiki.postgresql.org/wiki/Performance_Optimization
[2] https://use-the-index-luke.com/postgresql
[3] https://www.cybertec-postgresql.com/en/postgresql-query-optimization/"""
    },
    {
        "user": "What are the best practices for Docker in production?",
        "query": "Docker production best practices security",
        "results": [
            {"title": "Docker Production Best Practices", "url": "https://docs.docker.com/develop/dev-best-practices/", "snippet": "Use multi-stage builds. Don't run as root. Pin image versions."},
            {"title": "Docker Security", "url": "https://snyk.io/blog/docker-security-best-practices/", "snippet": "Scan images for vulnerabilities. Use read-only filesystems. Limit capabilities."},
            {"title": "Docker Performance", "url": "https://sysdig.com/blog/docker-performance/", "snippet": "Use slim base images. Layer caching. Health checks."}
        ],
        "report": """Here are Docker production best practices:

**Image Best Practices**
- Use multi-stage builds to minimize image size [1]
- Pin specific image versions (avoid `latest`) [1]
- Use slim or alpine base images [3]

**Security**
- Never run containers as root [1]
- Scan images for vulnerabilities [2]
- Use read-only root filesystems [2]
- Drop unnecessary Linux capabilities [2]

**Performance**
- Optimize layer caching in Dockerfiles [3]
- Implement health checks [3]

Sources:
[1] https://docs.docker.com/develop/dev-best-practices/
[2] https://snyk.io/blog/docker-security-best-practices/
[3] https://sysdig.com/blog/docker-performance/"""
    },
]

# =============================================================================
# Memory Scenarios - Extended
# =============================================================================

MEMORY_SCENARIOS = [
    {"user": "Remember that I prefer dark mode in all applications", "content": "User prefers dark mode interfaces", "category": "preferences"},
    {"user": "Save this: my project deadline is March 15th", "content": "Project deadline is March 15th", "category": "projects"},
    {"user": "Note that our API rate limit is 1000 requests per minute", "content": "API rate limit is 1000 requests per minute", "category": "technical"},
    {"user": "Remember my database password convention: service_env_random", "content": "Database password convention: service_env_random", "category": "conventions"},
    {"user": "Save that our staging server is at staging.example.com", "content": "Staging server URL: staging.example.com", "category": "infrastructure"},
    # New memory scenarios
    {"user": "Remember that I use VS Code as my primary editor", "content": "User's primary editor is VS Code", "category": "preferences"},
    {"user": "Note that our team standup is at 9:30 AM daily", "content": "Team standup meeting: 9:30 AM daily", "category": "schedule"},
    {"user": "Save that we use GitHub Actions for CI/CD", "content": "CI/CD platform: GitHub Actions", "category": "infrastructure"},
    {"user": "Remember that our production database is on AWS RDS", "content": "Production database: AWS RDS", "category": "infrastructure"},
    {"user": "Note that John is the lead on the payment system", "content": "John is the lead developer for payment system", "category": "team"},
    {"user": "Remember that we follow the Google Python style guide", "content": "Code style: Google Python style guide", "category": "conventions"},
    {"user": "Save that our error tracking uses Sentry", "content": "Error tracking platform: Sentry", "category": "infrastructure"},
    {"user": "Note that API v1 is deprecated, use v2", "content": "API v1 is deprecated, use v2 for all new integrations", "category": "technical"},
    {"user": "Remember that I prefer TypeScript over JavaScript", "content": "User prefers TypeScript over JavaScript", "category": "preferences"},
    {"user": "Save that our sprint length is 2 weeks", "content": "Sprint length: 2 weeks", "category": "process"},
]

RECALL_SCENARIOS = [
    {"user": "What do you remember about my preferences?", "query": "user preferences", "results": [
        {"id": "mem_1", "content": "User prefers dark mode interfaces", "category": "preferences", "relevance": 0.95},
        {"id": "mem_2", "content": "User prefers TypeScript over JavaScript", "category": "preferences", "relevance": 0.87},
    ]},
    {"user": "What do you know about our infrastructure?", "query": "infrastructure servers", "results": [
        {"id": "mem_3", "content": "Staging server: staging.example.com", "category": "infrastructure", "relevance": 0.92},
        {"id": "mem_4", "content": "Production database: AWS RDS", "category": "infrastructure", "relevance": 0.85},
    ]},
    # New recall scenarios
    {"user": "What conventions do we follow?", "query": "coding conventions style", "results": [
        {"id": "mem_5", "content": "Code style: Google Python style guide", "category": "conventions", "relevance": 0.93},
        {"id": "mem_6", "content": "Database password convention: service_env_random", "category": "conventions", "relevance": 0.82},
    ]},
    {"user": "What do you know about our team?", "query": "team members leads", "results": [
        {"id": "mem_7", "content": "John is the lead for payment system", "category": "team", "relevance": 0.91},
    ]},
    {"user": "What's our process for sprints?", "query": "sprint process agile", "results": [
        {"id": "mem_8", "content": "Sprint length: 2 weeks", "category": "process", "relevance": 0.89},
        {"id": "mem_9", "content": "Team standup: 9:30 AM daily", "category": "schedule", "relevance": 0.75},
    ]},
]

FACT_SCENARIOS = [
    {"user": "Note that project-alpha uses PostgreSQL", "subject": "project-alpha", "predicate": "uses_database", "object": "PostgreSQL"},
    {"user": "Record that the auth-service depends on Redis", "subject": "auth-service", "predicate": "depends_on", "object": "Redis"},
    {"user": "Save that John owns the payment-api", "subject": "John", "predicate": "owns", "object": "payment-api"},
    # New fact scenarios
    {"user": "Note that the frontend uses React", "subject": "frontend", "predicate": "built_with", "object": "React"},
    {"user": "Record that the API gateway routes to user-service", "subject": "api-gateway", "predicate": "routes_to", "object": "user-service"},
    {"user": "Save that production runs on Kubernetes", "subject": "production", "predicate": "runs_on", "object": "Kubernetes"},
    {"user": "Note that the notification-service uses RabbitMQ", "subject": "notification-service", "predicate": "uses", "object": "RabbitMQ"},
    {"user": "Record that Sarah maintains the data-pipeline", "subject": "Sarah", "predicate": "maintains", "object": "data-pipeline"},
    {"user": "Save that staging mirrors production", "subject": "staging", "predicate": "mirrors", "object": "production"},
    {"user": "Note that the mobile-app consumes the REST API", "subject": "mobile-app", "predicate": "consumes", "object": "REST API"},
]

# =============================================================================
# Reminder Scenarios - Extended
# =============================================================================

REMINDER_SCENARIOS = [
    {"user": "Remind me to review the PR in 2 hours", "content": "Review the PR", "due": "2 hours"},
    {"user": "Set a reminder for tomorrow to deploy to staging", "content": "Deploy to staging", "due": "tomorrow"},
    {"user": "Remind me in 30 minutes to check the build", "content": "Check the build", "due": "30 minutes"},
    {"user": "Create a reminder for Friday to submit the report", "content": "Submit the report", "due": "Friday"},
    {"user": "Remind me next week to follow up with the client", "content": "Follow up with client", "due": "1 week"},
    # New reminder scenarios
    {"user": "Set a reminder in 1 hour to take a break", "content": "Take a break", "due": "1 hour"},
    {"user": "Remind me at 3pm to join the meeting", "content": "Join the meeting", "due": "3pm"},
    {"user": "Create a reminder for Monday to review sprint goals", "content": "Review sprint goals", "due": "Monday"},
    {"user": "Remind me in 15 minutes to respond to that email", "content": "Respond to email", "due": "15 minutes"},
    {"user": "Set a reminder for end of day to push my changes", "content": "Push code changes", "due": "end of day"},
    {"user": "Remind me tomorrow morning to update the documentation", "content": "Update documentation", "due": "tomorrow morning"},
    {"user": "Create a reminder for next month to renew the SSL certificate", "content": "Renew SSL certificate", "due": "1 month"},
]

# =============================================================================
# Research/Scratchpad Scenarios - Extended
# =============================================================================

RESEARCH_SCENARIOS = [
    {
        "user": "I'm researching microservices patterns. Note that the Circuit Breaker pattern prevents cascade failures.",
        "topic": "microservices patterns",
        "content": "Circuit Breaker pattern prevents cascade failures by failing fast when a service is unavailable",
        "source": "Martin Fowler's blog",
        "confidence": "high",
        "tags": "microservices,patterns,resilience"
    },
    {
        "user": "Save this finding: GraphQL reduces over-fetching by letting clients specify exact data needs",
        "topic": "GraphQL benefits",
        "content": "GraphQL reduces over-fetching by allowing clients to specify exactly what data they need",
        "source": "GraphQL documentation",
        "confidence": "high",
        "tags": "graphql,api,performance"
    },
    # New research scenarios
    {
        "user": "Note that Redis Cluster provides automatic sharding across nodes",
        "topic": "Redis scaling",
        "content": "Redis Cluster automatically shards data across multiple nodes with automatic failover",
        "source": "Redis documentation",
        "confidence": "high",
        "tags": "redis,scaling,database"
    },
    {
        "user": "Save this: WebSockets maintain persistent connections unlike HTTP polling",
        "topic": "real-time communication",
        "content": "WebSockets provide full-duplex communication over a single persistent connection",
        "source": "MDN Web Docs",
        "confidence": "high",
        "tags": "websockets,real-time,networking"
    },
    {
        "user": "Record that CQRS separates read and write models for scalability",
        "topic": "architecture patterns",
        "content": "CQRS (Command Query Responsibility Segregation) uses separate models for reads and writes",
        "source": "Microsoft Architecture Guide",
        "confidence": "high",
        "tags": "cqrs,architecture,scalability"
    },
    {
        "user": "Note that JWT tokens should be short-lived with refresh tokens for security",
        "topic": "authentication",
        "content": "JWTs should have short expiration (15 min) with refresh tokens for longer sessions",
        "source": "OWASP Guidelines",
        "confidence": "high",
        "tags": "jwt,security,authentication"
    },
]

SCRATCHPAD_SCENARIOS = [
    {
        "user": "Save these requirements for later: 1) Must support OAuth, 2) Need rate limiting, 3) Should have webhooks",
        "key": "api_requirements",
        "content": "1. Must support OAuth 2.0\n2. Need rate limiting (1000 req/min)\n3. Should have webhook support"
    },
    # New scratchpad scenarios
    {
        "user": "Keep track of these API endpoints we need to build: /users, /posts, /comments",
        "key": "api_endpoints",
        "content": "Endpoints to build:\n- GET/POST /users\n- GET/POST /posts\n- GET/POST /comments"
    },
    {
        "user": "Save this list of dependencies we need: FastAPI, SQLAlchemy, Redis, Celery",
        "key": "project_dependencies",
        "content": "Dependencies:\n- FastAPI (web framework)\n- SQLAlchemy (ORM)\n- Redis (caching)\n- Celery (task queue)"
    },
    {
        "user": "Note down these database tables: users, posts, comments, likes",
        "key": "database_schema",
        "content": "Tables:\n- users (id, email, name, created_at)\n- posts (id, user_id, content, created_at)\n- comments (id, post_id, user_id, content)\n- likes (id, post_id, user_id)"
    },
    {
        "user": "Keep these environment variables: DATABASE_URL, REDIS_URL, SECRET_KEY, API_KEY",
        "key": "env_variables",
        "content": "Required env vars:\n- DATABASE_URL\n- REDIS_URL\n- SECRET_KEY\n- API_KEY"
    },
]

# =============================================================================
# Augmentation Templates
# =============================================================================

PLANNING_TEMPLATES = [
    "Help me plan {task}",
    "I need to plan {task}",
    "Create a plan for {task}",
    "Can you help me plan out {task}?",
    "I want to {task}, help me plan it",
    "Plan {task} for me",
    "What steps should I take to {task}?",
]

PLANNING_TASKS = [
    ("implementing a caching layer", "Caching Layer Implementation", "Add Redis caching to reduce database load"),
    ("setting up a testing pipeline", "Testing Pipeline Setup", "Implement comprehensive automated testing"),
    ("building a user dashboard", "User Dashboard Development", "Create an analytics dashboard for users"),
    ("implementing file uploads", "File Upload System", "Build secure file upload with S3 storage"),
    ("adding multi-tenancy", "Multi-tenancy Implementation", "Support multiple organizations in one instance"),
    ("setting up logging infrastructure", "Logging Infrastructure", "Implement centralized logging and monitoring"),
    ("building an admin panel", "Admin Panel Development", "Create administrative interface for management"),
    ("implementing webhooks", "Webhook System", "Build outgoing webhook notifications"),
    ("adding two-factor authentication", "2FA Implementation", "Add two-factor authentication for security"),
    ("setting up a CDN", "CDN Configuration", "Configure CDN for static asset delivery"),
    ("implementing API pagination", "API Pagination", "Add cursor-based pagination to API endpoints"),
    ("building a job queue", "Background Job System", "Implement async job processing with Celery/Redis"),
    ("adding email notifications", "Email Notification System", "Build transactional email system"),
    ("implementing data exports", "Data Export Feature", "Allow users to export their data"),
    ("setting up feature flags", "Feature Flag System", "Implement feature flags for gradual rollouts"),
]

SEARCH_QUERIES = [
    ("best practices for {topic}", "{topic} best practices 2024"),
    ("how to implement {topic}", "{topic} implementation guide tutorial"),
    ("what is {topic}", "{topic} explained overview"),
    ("{topic} vs alternatives", "{topic} comparison alternatives 2024"),
    ("common {topic} mistakes", "{topic} pitfalls mistakes avoid"),
    ("{topic} security considerations", "{topic} security best practices"),
    ("optimizing {topic}", "{topic} optimization performance tips"),
    ("{topic} in production", "{topic} production deployment guide"),
]

SEARCH_TOPICS = [
    "database indexing", "API versioning", "container orchestration",
    "message queues", "load balancing", "caching strategies",
    "authentication", "logging", "monitoring", "CI/CD pipelines",
    "serverless architecture", "microservices", "GraphQL",
    "WebSockets", "rate limiting", "database migrations",
    "error handling", "testing strategies", "code review",
    "git workflows", "Docker optimization", "Kubernetes networking",
]

MEMORY_TEMPLATES = [
    "Remember that {fact}",
    "Save this: {fact}",
    "Note that {fact}",
    "Keep in mind that {fact}",
    "Store this information: {fact}",
    "Don't forget that {fact}",
]

MEMORY_FACTS = [
    ("I prefer tabs over spaces", "preferences"),
    ("our team uses Slack for communication", "tools"),
    ("deployment happens every Tuesday", "process"),
    ("the API documentation is at docs.example.com", "infrastructure"),
    ("we use pytest for testing", "conventions"),
    ("maximum file upload size is 10MB", "technical"),
    ("the primary language is Python", "conventions"),
    ("we follow semantic versioning", "conventions"),
    ("backups run at 2 AM UTC", "infrastructure"),
    ("code reviews require 2 approvals", "process"),
]

REMINDER_TEMPLATES = [
    "Remind me {timeframe} to {task}",
    "Set a reminder {timeframe} to {task}",
    "Create a reminder {timeframe} to {task}",
    "In {timeframe}, remind me to {task}",
    "Don't let me forget to {task} {timeframe}",
]

REMINDER_TASKS = [
    "check the server logs",
    "merge the feature branch",
    "update the dependencies",
    "run the performance tests",
    "backup the database",
    "review the security alerts",
    "check the monitoring dashboard",
    "update the API documentation",
    "respond to the code review",
    "test the new deployment",
]

REMINDER_TIMEFRAMES = [
    "in 30 minutes", "in 1 hour", "in 2 hours",
    "tomorrow", "tomorrow morning", "this afternoon",
    "next Monday", "next week", "in 3 days",
]

# =============================================================================
# Generation Functions
# =============================================================================

def generate_planning_examples() -> List[Dict[str, Any]]:
    """Generate planning tool training examples."""
    examples = []

    for scenario in PLANNING_SCENARIOS:
        assistant_content = f"""I'll create a structured plan for this project.

{tool_call("create_plan", scenario["plan"])}"""

        tool_result = json.dumps({
            "success": True,
            "plan_id": random.randint(1, 100),
            "title": scenario["plan"]["title"],
            "steps": len(scenario["plan"]["steps"])
        })

        final_response = f"""I've created a plan with {len(scenario["plan"]["steps"])} steps:

**{scenario["plan"]["title"]}**
Goal: {scenario["plan"]["goal"]}

Steps:
""" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(scenario["plan"]["steps"]))

        examples.append({
            "type": "multi_turn",
            "conversation": [
                {"role": "user", "content": scenario["user"]},
                {"role": "assistant", "content": assistant_content},
                {"role": "tool_result", "content": tool_result},
                {"role": "assistant", "content": final_response}
            ]
        })

    for scenario in THINK_STEP_BY_STEP_SCENARIOS:
        params = {
            "problem": scenario["problem"],
            "steps": scenario["steps"],
            "conclusion": scenario["conclusion"],
            "confidence": scenario["confidence"]
        }
        if "caveats" in scenario:
            params["caveats"] = scenario["caveats"]

        assistant_content = f"""Let me think through this systematically.

{tool_call("think_step_by_step", params)}"""

        tool_result = json.dumps({
            "problem": scenario["problem"],
            "conclusion": scenario["conclusion"],
            "confidence": scenario["confidence"]
        })

        final_response = f"""After analyzing this step by step:

**Conclusion:** {scenario["conclusion"]}
**Confidence:** {scenario["confidence"]}

Key reasoning:
""" + "\n".join(f"- {step}" for step in scenario["steps"][:3])

        examples.append({
            "type": "multi_turn",
            "conversation": [
                {"role": "user", "content": scenario["user"]},
                {"role": "assistant", "content": assistant_content},
                {"role": "tool_result", "content": tool_result},
                {"role": "assistant", "content": final_response}
            ]
        })

    for scenario in COMPARE_OPTIONS_SCENARIOS:
        params = {
            "title": scenario["title"],
            "options": scenario["options"],
            "criteria": scenario["criteria"],
            "analysis": scenario["analysis"],
            "recommendation": scenario["recommendation"]
        }

        assistant_content = f"""I'll create a structured comparison.

{tool_call("compare_options", params)}"""

        tool_result = json.dumps({
            "comparison_id": random.randint(1, 100),
            "title": scenario["title"]
        })

        table_header = "| Criteria | " + " | ".join(scenario["options"]) + " |"
        table_sep = "|" + "|".join(["---"] * (len(scenario["options"]) + 1)) + "|"
        table_rows = []
        for criterion in scenario["criteria"]:
            row = f"| {criterion} |"
            for opt in scenario["options"]:
                row += f" {scenario['analysis'][opt][criterion]} |"
            table_rows.append(row)

        final_response = f"""Here's my comparison:

{table_header}
{table_sep}
""" + "\n".join(table_rows) + f"""

**Recommendation:** {scenario["recommendation"]}"""

        examples.append({
            "type": "multi_turn",
            "conversation": [
                {"role": "user", "content": scenario["user"]},
                {"role": "assistant", "content": assistant_content},
                {"role": "tool_result", "content": tool_result},
                {"role": "assistant", "content": final_response}
            ]
        })

    return examples


def generate_web_search_examples() -> List[Dict[str, Any]]:
    """Generate web search examples with citations."""
    examples = []

    for scenario in WEB_SEARCH_SCENARIOS:
        assistant_content = f"""I'll search for current information on this topic.

{tool_call("web_search", {"query": scenario["query"], "num_results": 5})}"""

        tool_result = json.dumps({
            "query": scenario["query"],
            "num_results": len(scenario["results"]),
            "results": scenario["results"]
        })

        examples.append({
            "type": "multi_turn",
            "conversation": [
                {"role": "user", "content": scenario["user"]},
                {"role": "assistant", "content": assistant_content},
                {"role": "tool_result", "content": tool_result},
                {"role": "assistant", "content": scenario["report"]}
            ]
        })

    return examples


def generate_memory_examples() -> List[Dict[str, Any]]:
    """Generate memory and knowledge graph examples."""
    examples = []

    for scenario in MEMORY_SCENARIOS:
        assistant_content = f"""I'll save that to memory.

{tool_call("save_memory", {"content": scenario["content"], "category": scenario["category"]})}"""

        tool_result = json.dumps({"success": True, "memory_id": f"mem_{random.randint(100, 999)}"})

        examples.append({
            "type": "multi_turn",
            "conversation": [
                {"role": "user", "content": scenario["user"]},
                {"role": "assistant", "content": assistant_content},
                {"role": "tool_result", "content": tool_result},
                {"role": "assistant", "content": f"I've saved that to my memory under '{scenario['category']}'."}
            ]
        })

    for scenario in RECALL_SCENARIOS:
        assistant_content = f"""Let me search my memory.

{tool_call("recall", {"query": scenario["query"], "num_results": 5})}"""

        tool_result = json.dumps(scenario["results"])
        memories_text = "\n".join(f"- {r['content']}" for r in scenario["results"])

        examples.append({
            "type": "multi_turn",
            "conversation": [
                {"role": "user", "content": scenario["user"]},
                {"role": "assistant", "content": assistant_content},
                {"role": "tool_result", "content": tool_result},
                {"role": "assistant", "content": f"Here's what I remember:\n\n{memories_text}"}
            ]
        })

    for scenario in FACT_SCENARIOS:
        assistant_content = f"""I'll save that as a fact.

{tool_call("save_fact", {"subject": scenario["subject"], "predicate": scenario["predicate"], "object": scenario["object"]})}"""

        tool_result = json.dumps(scenario)

        examples.append({
            "type": "multi_turn",
            "conversation": [
                {"role": "user", "content": scenario["user"]},
                {"role": "assistant", "content": assistant_content},
                {"role": "tool_result", "content": tool_result},
                {"role": "assistant", "content": f"Saved: {scenario['subject']} → {scenario['predicate']} → {scenario['object']}"}
            ]
        })

    return examples


def generate_reminder_examples() -> List[Dict[str, Any]]:
    """Generate reminder tool examples."""
    examples = []

    for scenario in REMINDER_SCENARIOS:
        assistant_content = f"""I'll create that reminder.

{tool_call("create_reminder", {"content": scenario["content"], "due_at": scenario["due"]})}"""

        tool_result = json.dumps({
            "reminder_id": random.randint(1, 100),
            "content": scenario["content"],
            "due_at": scenario["due"],
            "created": True
        })

        examples.append({
            "type": "multi_turn",
            "conversation": [
                {"role": "user", "content": scenario["user"]},
                {"role": "assistant", "content": assistant_content},
                {"role": "tool_result", "content": tool_result},
                {"role": "assistant", "content": f"Done! I've set a reminder to \"{scenario['content']}\" for {scenario['due']}."}
            ]
        })

    # list_reminders
    examples.append({
        "type": "multi_turn",
        "conversation": [
            {"role": "user", "content": "What reminders do I have?"},
            {"role": "assistant", "content": f"Let me check your reminders.\n\n{tool_call('list_reminders', {})}"},
            {"role": "tool_result", "content": json.dumps([
                {"id": 1, "content": "Review the PR", "due_at": "2024-01-15 14:00"},
                {"id": 2, "content": "Deploy to staging", "due_at": "2024-01-16 09:00"}
            ])},
            {"role": "assistant", "content": "You have 2 pending reminders:\n\n1. **Review the PR** - Due: Jan 15, 2:00 PM\n2. **Deploy to staging** - Due: Jan 16, 9:00 AM"}
        ]
    })

    return examples


def generate_research_examples() -> List[Dict[str, Any]]:
    """Generate research and scratchpad examples."""
    examples = []

    for scenario in RESEARCH_SCENARIOS:
        params = {
            "topic": scenario["topic"],
            "content": scenario["content"],
            "source": scenario["source"],
            "confidence": scenario["confidence"],
            "tags": scenario["tags"]
        }

        assistant_content = f"""I'll save that research finding.

{tool_call("research_note", params)}"""

        tool_result = json.dumps({"note_id": random.randint(1, 100), "saved": True})

        examples.append({
            "type": "multi_turn",
            "conversation": [
                {"role": "user", "content": scenario["user"]},
                {"role": "assistant", "content": assistant_content},
                {"role": "tool_result", "content": tool_result},
                {"role": "assistant", "content": f"Saved research note on '{scenario['topic']}'."}
            ]
        })

    for scenario in SCRATCHPAD_SCENARIOS:
        assistant_content = f"""I'll save that to my scratchpad.

{tool_call("scratchpad_write", {"key": scenario["key"], "content": scenario["content"]})}"""

        tool_result = json.dumps({"key": scenario["key"], "saved": True})

        examples.append({
            "type": "multi_turn",
            "conversation": [
                {"role": "user", "content": scenario["user"]},
                {"role": "assistant", "content": assistant_content},
                {"role": "tool_result", "content": tool_result},
                {"role": "assistant", "content": f"Saved to scratchpad under '{scenario['key']}'."}
            ]
        })

    return examples


def generate_workflow_examples() -> List[Dict[str, Any]]:
    """Generate multi-tool workflow examples."""
    examples = []

    # Research + Planning
    examples.append({
        "type": "multi_turn",
        "conversation": [
            {"role": "user", "content": "Help me research and plan implementing a caching layer"},
            {"role": "assistant", "content": f"""I'll research caching best practices first.

{tool_call("web_search", {"query": "API caching best practices Redis 2024", "num_results": 5})}"""},
            {"role": "tool_result", "content": json.dumps({
                "results": [
                    {"title": "Caching Strategies", "url": "https://aws.amazon.com/caching/", "snippet": "Cache-aside pattern. Set appropriate TTLs."},
                    {"title": "Redis vs Memcached", "url": "https://redis.io/compare/", "snippet": "Redis for complex data structures."}
                ]
            })},
            {"role": "assistant", "content": f"""Good findings. Now let me create a plan.

{tool_call("create_plan", {
    "title": "Caching Implementation",
    "goal": "Add Redis caching to reduce database load",
    "steps": ["Set up Redis", "Implement cache-aside pattern", "Add caching to endpoints", "Configure TTLs", "Add monitoring"]
})}"""},
            {"role": "tool_result", "content": json.dumps({"plan_id": 1, "steps": 5})},
            {"role": "assistant", "content": """Based on my research and planning:

**Key Findings:**
- Use cache-aside pattern for flexibility [1]
- Redis preferred for complex data structures [2]

**Plan:** 5 steps to implement caching, starting with Redis setup.

Sources:
[1] https://aws.amazon.com/caching/
[2] https://redis.io/compare/"""}
        ]
    })

    # Research + Save findings
    examples.append({
        "type": "multi_turn",
        "conversation": [
            {"role": "user", "content": "Research Kubernetes autoscaling and save key findings"},
            {"role": "assistant", "content": f"""I'll research K8s autoscaling.

{tool_call("web_search", {"query": "Kubernetes autoscaling HPA VPA best practices", "num_results": 5})}"""},
            {"role": "tool_result", "content": json.dumps({
                "results": [
                    {"title": "K8s Autoscaling", "url": "https://kubernetes.io/docs/hpa/", "snippet": "HPA scales pods based on metrics."},
                ]
            })},
            {"role": "assistant", "content": f"""Found useful info. Let me save it.

{tool_call("research_note", {
    "topic": "Kubernetes Autoscaling",
    "content": "HPA for replica scaling, VPA for resource adjustment",
    "source": "kubernetes.io",
    "confidence": "high",
    "tags": "kubernetes,autoscaling"
})}"""},
            {"role": "tool_result", "content": json.dumps({"note_id": 15, "saved": True})},
            {"role": "assistant", "content": """Here's what I found about K8s autoscaling:

**HPA (Horizontal Pod Autoscaler)** - Scales pod replicas based on CPU/memory

I've saved these findings for future reference.

Source: https://kubernetes.io/docs/hpa/"""}
        ]
    })

    return examples


def augment_planning_examples(count: int) -> List[Dict[str, Any]]:
    """Generate augmented planning examples."""
    examples = []

    for _ in range(count):
        task_info = random.choice(PLANNING_TASKS)
        template = random.choice(PLANNING_TEMPLATES)
        user_msg = template.format(task=task_info[0])

        steps = [
            f"Define requirements for {task_info[0]}",
            "Set up project structure",
            "Implement core functionality",
            "Add error handling",
            "Write tests",
            "Deploy and monitor"
        ]

        plan = {
            "title": task_info[1],
            "goal": task_info[2],
            "steps": steps
        }

        assistant_content = f"""I'll create a plan for this.

{tool_call("create_plan", plan)}"""

        tool_result = json.dumps({"success": True, "plan_id": random.randint(1, 100), "steps": len(steps)})

        examples.append({
            "type": "multi_turn",
            "conversation": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_content},
                {"role": "tool_result", "content": tool_result},
                {"role": "assistant", "content": f"I've created a plan for {task_info[0]} with {len(steps)} steps."}
            ]
        })

    return examples


def augment_web_search_examples(count: int) -> List[Dict[str, Any]]:
    """Generate augmented web search examples."""
    examples = []

    for _ in range(count):
        topic = random.choice(SEARCH_TOPICS)
        query_template = random.choice(SEARCH_QUERIES)
        user_msg = query_template[0].format(topic=topic)
        query = query_template[1].format(topic=topic)

        results = [
            {"title": f"{topic.title()} Guide", "url": f"https://example.com/{topic.replace(' ', '-')}", "snippet": f"Comprehensive guide to {topic}."},
            {"title": f"Best Practices for {topic.title()}", "url": f"https://docs.example.com/{topic.replace(' ', '-')}", "snippet": f"Industry best practices for {topic}."},
        ]

        assistant_content = f"""I'll search for information on this.

{tool_call("web_search", {"query": query, "num_results": 5})}"""

        tool_result = json.dumps({"query": query, "results": results})

        report = f"""Here's what I found about {topic}:

**Key Points:**
- Comprehensive guide available [1]
- Industry best practices documented [2]

Sources:
[1] https://example.com/{topic.replace(' ', '-')}
[2] https://docs.example.com/{topic.replace(' ', '-')}"""

        examples.append({
            "type": "multi_turn",
            "conversation": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_content},
                {"role": "tool_result", "content": tool_result},
                {"role": "assistant", "content": report}
            ]
        })

    return examples


def augment_memory_examples(count: int) -> List[Dict[str, Any]]:
    """Generate augmented memory examples."""
    examples = []

    for _ in range(count):
        fact_info = random.choice(MEMORY_FACTS)
        template = random.choice(MEMORY_TEMPLATES)
        user_msg = template.format(fact=fact_info[0])

        assistant_content = f"""I'll save that to memory.

{tool_call("save_memory", {"content": fact_info[0], "category": fact_info[1]})}"""

        tool_result = json.dumps({"success": True, "memory_id": f"mem_{random.randint(100, 999)}"})

        examples.append({
            "type": "multi_turn",
            "conversation": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_content},
                {"role": "tool_result", "content": tool_result},
                {"role": "assistant", "content": f"I've saved that to my memory under '{fact_info[1]}'."}
            ]
        })

    return examples


def augment_reminder_examples(count: int) -> List[Dict[str, Any]]:
    """Generate augmented reminder examples."""
    examples = []

    for _ in range(count):
        task = random.choice(REMINDER_TASKS)
        timeframe = random.choice(REMINDER_TIMEFRAMES)
        template = random.choice(REMINDER_TEMPLATES)
        user_msg = template.format(task=task, timeframe=timeframe)

        assistant_content = f"""I'll set that reminder.

{tool_call("create_reminder", {"content": task.capitalize(), "due_at": timeframe})}"""

        tool_result = json.dumps({
            "reminder_id": random.randint(1, 100),
            "content": task.capitalize(),
            "due_at": timeframe,
            "created": True
        })

        examples.append({
            "type": "multi_turn",
            "conversation": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_content},
                {"role": "tool_result", "content": tool_result},
                {"role": "assistant", "content": f"Done! Reminder set to {task} {timeframe}."}
            ]
        })

    return examples


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate AI-Assistant training data")
    parser.add_argument('--output', type=str, default='/root/qwen-server/training-data/assistant-tools.jsonl')
    parser.add_argument('--augment-planning', type=int, default=200, help="Augmented planning examples")
    parser.add_argument('--augment-search', type=int, default=150, help="Augmented search examples")
    parser.add_argument('--augment-memory', type=int, default=100, help="Augmented memory examples")
    parser.add_argument('--augment-reminders', type=int, default=50, help="Augmented reminder examples")
    args = parser.parse_args()

    print("Generating AI-Assistant training data...")
    print("=" * 60)

    all_examples = []

    # Base examples from scenarios
    planning = generate_planning_examples()
    print(f"Base planning examples: {len(planning)}")
    all_examples.extend(planning)

    web = generate_web_search_examples()
    print(f"Base web search examples: {len(web)}")
    all_examples.extend(web)

    memory = generate_memory_examples()
    print(f"Base memory examples: {len(memory)}")
    all_examples.extend(memory)

    reminders = generate_reminder_examples()
    print(f"Base reminder examples: {len(reminders)}")
    all_examples.extend(reminders)

    research = generate_research_examples()
    print(f"Base research examples: {len(research)}")
    all_examples.extend(research)

    workflows = generate_workflow_examples()
    print(f"Base workflow examples: {len(workflows)}")
    all_examples.extend(workflows)

    print("-" * 60)

    # Augmented examples
    aug_planning = augment_planning_examples(args.augment_planning)
    print(f"Augmented planning: {len(aug_planning)}")
    all_examples.extend(aug_planning)

    aug_search = augment_web_search_examples(args.augment_search)
    print(f"Augmented search: {len(aug_search)}")
    all_examples.extend(aug_search)

    aug_memory = augment_memory_examples(args.augment_memory)
    print(f"Augmented memory: {len(aug_memory)}")
    all_examples.extend(aug_memory)

    aug_reminders = augment_reminder_examples(args.augment_reminders)
    print(f"Augmented reminders: {len(aug_reminders)}")
    all_examples.extend(aug_reminders)

    # Shuffle
    random.shuffle(all_examples)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    print("=" * 60)
    print(f"Total examples: {len(all_examples)}")
    print(f"Saved to: {output_path}")

    # Category breakdown
    print("\nCategory distribution:")
    print(f"  Planning/Reasoning: {len(planning) + len(aug_planning)}")
    print(f"  Web Search: {len(web) + len(aug_search)}")
    print(f"  Memory/Facts: {len(memory) + len(aug_memory)}")
    print(f"  Reminders: {len(reminders) + len(aug_reminders)}")
    print(f"  Research: {len(research)}")
    print(f"  Workflows: {len(workflows)}")


if __name__ == "__main__":
    main()
