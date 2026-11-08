# StrataFi's web app for compose's "app" profile: the production build, served by
# `vite preview`, which proxies /api to the api service. For one machine, not for the internet.
FROM node:24-bookworm-slim@sha256:d6aa754f16b3197301076f047b5def2f02ea1dbbc2ca920407d46d7ec7f87b20

LABEL org.opencontainers.image.title="stratafi-web" \
      org.opencontainers.image.description="StrataFi, a lab for evaluating RAG pipelines: its web app" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /opt/stratafi/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web ./
RUN npm run build && chown -R node:node /opt/stratafi/web
USER node
ENV STRATAFI_API_URL=http://api:8000
# Not 4190: browsers refuse it, as the Fetch standard's list of blocked ports says.
EXPOSE 4180
CMD ["npx", "vite", "preview", "--host", "0.0.0.0", "--port", "4180", "--strictPort"]
