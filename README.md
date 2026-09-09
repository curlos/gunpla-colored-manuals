# Gunpla Colored Manuals

Colorizes black-and-white Gunpla instruction manuals into the kit's actual molded colors, and tracks used vs. unused parts per kit — using data already printed in the manuals (runner color labels, "X" marks for unused parts).

## Structure (monorepo)

- [`frontend/`](frontend/) — React + TypeScript + Vite + RTK Query
- [`backend/`](backend/) — Node.js + TypeScript + GraphQL live API (PostgreSQL, S3)
- [`pipeline/`](pipeline/) — Python processing pipeline (scraping + CV/OCR manual analysis), run as a background job

## Stack

PostgreSQL · GraphQL · Docker · AWS (ECS/Fargate, RDS, S3)
