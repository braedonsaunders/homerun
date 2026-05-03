# syntax=docker/dockerfile:1.7
#
# Homerun frontend image — builds the Vite/React app and serves the
# static bundle via nginx, with /api and /ws reverse-proxied to the
# backend container.  The frontend uses relative paths
# (axios.create({ baseURL: '/api' }), see services/apiClient.ts), so
# everything is same-origin behind nginx.

ARG NODE_VERSION=20
ARG NGINX_VERSION=1.27-alpine

# ---------- builder ----------------------------------------------------------
FROM node:${NODE_VERSION}-alpine AS builder

WORKDIR /build

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ---------- runtime ----------------------------------------------------------
FROM nginx:${NGINX_VERSION} AS runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /build/dist /usr/share/nginx/html

EXPOSE 3000

# nginx:alpine ships its own entrypoint that handles signals correctly.
