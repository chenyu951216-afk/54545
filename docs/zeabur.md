# Zeabur Deployment

## Plain Explanation

Do not deploy this project by repeatedly clicking `Add files via upload`.

That mode can leave the container with only part of the project, which causes errors such as:

- `No module named 'tw_stock_ai.adapters'`
- `No module named 'tw_stock_ai.worker'`

Recommended production deployment:

1. One PostgreSQL service
2. One `web` service from the GitHub repository
3. One `worker` service from the same GitHub repository

The repository is the same.
The startup role can now be selected with environment variables:

- `SERVICE_ROLE=web`
- `SERVICE_ROLE=worker`

If Zeabur does not expose a Dockerfile selector for GitHub deployments, both services can still use the same root `Dockerfile`.

## Official Zeabur Notes

Based on Zeabur official docs:

- Zeabur supports deploying directly from a GitHub repository
- Zeabur supports Dockerfile-based deployment
- named Dockerfiles are supported for service-specific builds
- environment variables can reference exposed variables such as PostgreSQL connection strings

Sources:

- [Deploying with Dockerfile](https://zeabur.com/docs/en-US/deploy/dockerfile)
- [Deployment Methods](https://zeabur.com/docs/en-US/deploy/methods)
- [Create Service](https://zeabur.com/docs/en-US/deploy/create-service)
- [Setting Environment Variables](https://zeabur.com/docs/en-US/deploy/config/environment-variables)
- [PostgreSQL Deploy Guide](https://zeabur.com/templates/773OAW)

## Files in This Repo

- `Dockerfile`
  - deploy target for `web`
- `worker.Dockerfile`
  - deploy target for `worker`
- `Dockerfile.worker`
  - compatibility alias for `worker`
- `docs/zeabur.web.env.example`
  - full environment variables for `web`
- `docs/zeabur.worker.env.example`
  - full environment variables for `worker`
- `.dockerignore`
  - keeps temporary files, local databases, and secrets out of Docker build context

## Step-By-Step Deployment

### 1. Push This Project to GitHub

Make sure the repository contains the full project root:

- `src/`
- `tests/`
- `Dockerfile`
- `worker.Dockerfile`
- `pyproject.toml`
- `requirements.txt`
- `docs/`

Do not rely on upload-only deployment for this project.

### 2. Create PostgreSQL in Zeabur

In Zeabur:

1. click `建立服務`
2. choose `Databases`
3. choose `PostgreSQL`
4. wait until the database service is ready

Then open that PostgreSQL service and find either:

- the full connection string in `Instruction` / `Connection`

or these exposed values:

- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `POSTGRES_DATABASE`
- `POSTGRES_USERNAME`
- `POSTGRES_PASSWORD`

### 3. Create the `web` Service

In Zeabur:

1. click `建立服務`
2. choose `GitHub`
3. select this repository
4. name the service `web`
5. let Zeabur build using the root `Dockerfile`
6. paste the environment variables from `docs/zeabur.web.env.example`

Important:

- `SERVICE_ROLE=web`
- `ENABLE_SCHEDULER=false`
- this service is the website and API
- do not manually set `PORT`; Zeabur injects it automatically

### 4. Create the `worker` Service

In Zeabur:

1. click `建立服務`
2. choose `GitHub`
3. select the same repository
4. name the service `worker`
5. configure this service to use `worker.Dockerfile` or `Dockerfile.worker`
6. paste the environment variables from `docs/zeabur.worker.env.example`

Important:

- `SERVICE_ROLE=worker`
- `ENABLE_SCHEDULER=true`
- this service is the background scheduler
- it is responsible for:
  - startup bootstrap refresh
  - 07:00 prewarm
  - 08:00 daily push
- do not manually set `PORT`; Zeabur injects it automatically

### 5. Fill the Required Secrets

You must replace these placeholders in both services:

- `DATABASE_URL`
- `OPENAI_API_KEY`
- `FUGLE_API_KEY`
- `DISCORD_WEBHOOK_URL`

Optional but recommended:

- `FINMIND_API_TOKEN`

### 6. Deploy and Verify

After both services are created:

1. deploy `web`
2. deploy `worker`
3. open the `web` domain
4. verify:
   - `/api/health`
   - `/system`
   - `/picks`

Then check `worker` logs for:

- startup bootstrap started
- market data refresh completed
- prewarm job scheduled

## Correct DATABASE_URL on Zeabur

Best practice:

- paste the actual PostgreSQL connection string into `DATABASE_URL`
- do not manually create a `PORT` variable in Zeabur for this project

Example format:

```env
DATABASE_URL=postgresql+psycopg://username:password@host:5432/database
```

Do not leave placeholders such as:

- `<host>`
- `<port>`
- `REPLACE_WITH_ACTUAL_POSTGRESQL_PSYCOPG_URL`

If your Zeabur project reliably exposes `POSTGRES_CONNECTION_STRING`, you may use:

```env
DATABASE_URL=${POSTGRES_CONNECTION_STRING}
```

But if variable expansion is inconsistent in your service, paste the full real string directly.

The app also accepts:

- `postgres://...`
- `postgresql://...`
- `postgresql+psycopg://...`

## Environment Variable Split

### Common to Both `web` and `worker`

- `DATABASE_URL`
- `OPENAI_API_KEY`
- `FUGLE_API_KEY`
- `DISCORD_WEBHOOK_URL`
- providers and strategy settings

### Different Between `web` and `worker`

`web`:

- `SERVICE_ROLE=web`
- `ENABLE_SCHEDULER=false`

`worker`:

- `SERVICE_ROLE=worker`
- `ENABLE_SCHEDULER=true`

## Why Two Services Are Still Required

The repository is the same.
The runtime job is different.

`web` serves pages and APIs.

`worker` runs background jobs and does not need public traffic.

Do not merge them into a single production service if you want reliable 07:00 / 08:00 scheduling.

## What the Worker Does

On startup:

- optional startup bootstrap refresh

At 07:00 Asia/Taipei:

- refreshes real market data
- reruns screening
- performs AI analysis for candidate symbols and holdings
- prepares the daily report

At 08:00 Asia/Taipei:

- sends the prepared daily report to Discord

## If You Still Want to Use Upload

Not recommended.

If you must use upload:

- upload the entire project root in one shot
- do not upload files incrementally
- do not delete partial paths between uploads
- if you are uploading from Windows, generate a fresh bundle first with:
  - `powershell -ExecutionPolicy Bypass -File .\scripts\create_zeabur_upload_bundle.ps1`
  - then upload the generated `local_artifacts\zeabur_upload_bundle` folder instead of cherry-picking files
- if Zeabur upload mode still loses nested folders, use the wheel bundle instead:
  - `powershell -ExecutionPolicy Bypass -File .\scripts\create_zeabur_wheel_bundle.ps1`
  - upload the generated `_zeabur_wheel_bundle` folder
  - this mode installs the app from a root-level `.whl` file, so it does not depend on `src/tw_stock_ai/adapters` being individually uploaded
- if Windows packaging tools are unstable in your local environment, use the archive bundle:
  - `powershell -ExecutionPolicy Bypass -File .\scripts\create_zeabur_archive_bundle.ps1`
  - upload the generated `local_artifacts\zeabur_archive_bundle` folder
  - this mode uploads a root-level `app_bundle.zip`, and the Dockerfiles auto-extract it during build
- if Zeabur upload still drops `.zip` files from the root, use the embedded bundle:
  - `powershell -ExecutionPolicy Bypass -File .\scripts\create_zeabur_embedded_bundle.ps1`
  - upload the generated `local_artifacts\zeabur_embedded_bundle` folder
  - this mode reconstructs the full source tree from a root-level Python payload file instead of relying on a `.zip` upload
- if upload mode keeps deploying the worker service with the web entrypoint, generate service-specific bundles:
  - `powershell -ExecutionPolicy Bypass -File .\scripts\create_zeabur_service_bundle.ps1 -Service web`
  - `powershell -ExecutionPolicy Bypass -File .\scripts\create_zeabur_service_bundle.ps1 -Service worker`
  - upload `local_artifacts\zeabur_web_bundle` only to the web service
  - upload `local_artifacts\zeabur_worker_bundle` only to the worker service
  - the worker bundle rewrites the root `Dockerfile` to the worker entrypoint, so Zeabur upload mode cannot accidentally boot Uvicorn for the worker service
- if you deploy from GitHub and Zeabur does not let you choose a different Dockerfile for the worker service, set `SERVICE_ROLE=worker` on that service and keep using the root `Dockerfile`

Otherwise Zeabur may build a container missing:

- `tw_stock_ai.adapters`
- `tw_stock_ai.worker`
- templates
- prompts

The failure mode is now intentionally easier to diagnose.
If build logs show an error like:

- `COPY src/tw_stock_ai/adapters /app/src/tw_stock_ai/adapters: not found`

that means the upload bundle itself is incomplete before Python installation even starts.
In that case, fix the uploaded file tree rather than changing application logic.

## Logs

- application logs are written to `data/logs/`
- on Zeabur, prefer platform runtime logs first
- if `web` fails during build, check build logs
- if `worker` fails after start, check runtime logs
- if `web` runtime logs show `web_startup ... runtime_port_env=None` and Uvicorn still binds to `8000`, a 502 usually means the service port does not match what Zeabur is routing
- the `web` service should bind to `${PORT}` and normally falls back to `8080`
- the `worker` service should log `worker_started ...`; if you see Uvicorn instead, that service is using the wrong Dockerfile or startup target
