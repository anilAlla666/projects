# CIPHER container image — drop-in driver-level GPU efficiency layer.
# Mount or COPY into the customer's container, run via /opt/cipher/bin/cipher_run.sh.
FROM ubuntu:22.04

LABEL org.opencontainers.image.title="CIPHER"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.description="LD_PRELOAD GPU efficiency middleware"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 ca-certificates openssl python3 python3-minimal \
 && rm -rf /var/lib/apt/lists/*

# Layout matches the .deb install paths.
COPY libcipher_hook.so      /opt/cipher/lib/
COPY libcipher_rt.so        /opt/cipher/lib/
COPY libcipher_nccl_tuner.so /opt/cipher/lib/
COPY libnccl-tuner-cipher.so /opt/cipher/lib/
COPY cipher_run.sh          /opt/cipher/bin/
COPY cipher_python          /opt/cipher/lib/cipher_python
COPY policy.json.example    /opt/cipher/etc/

RUN chmod +x /opt/cipher/bin/cipher_run.sh \
 && ln -sf /opt/cipher/bin/cipher_run.sh /usr/local/bin/cipher_run.sh \
 && install -d -m 0755 /etc/cipher /var/log/cipher \
 && cp /opt/cipher/etc/policy.json.example /etc/cipher/policy.json \
 && umask 077 && openssl rand -hex 32 > /etc/cipher/hmac.key

ENV CIPHER_RT_PATH=/opt/cipher/lib/libcipher_rt.so
ENV CIPHER_PROFILE=production
ENV CIPHER_POLICY_PATH=/etc/cipher/policy.json

# Default entrypoint runs the customer's command under cipher_run.sh.
ENTRYPOINT ["/opt/cipher/bin/cipher_run.sh"]
CMD ["python3", "-c", "import torch; print('CIPHER container ready, torch:', torch.__version__)"]
