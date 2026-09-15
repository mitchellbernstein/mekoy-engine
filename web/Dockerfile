# The portal. Separate from the control plane because they are different runtimes: the
# API is Python, the portal is Node. A self-hoster runs both from the compose file at the
# repository root.
FROM node:24-slim AS deps

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund


FROM node:24-slim AS build

WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# The portal reads its settings at runtime, but Next inlines `NEXT_PUBLIC_*` at build
# time, so anything public has to be present here rather than in the compose file.
RUN npm run build


FROM node:24-slim AS run

WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/.next ./.next
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/package.json ./package.json
COPY --from=build /app/public ./public
COPY --from=build /app/next.config.ts ./next.config.ts

# The API this portal talks to. Compose overrides it with the service name; the default
# is right for running the container alone against a control plane on the host.
ENV MEKOY_API_URL=http://127.0.0.1:8787

EXPOSE 3000
CMD ["npm", "run", "start"]
