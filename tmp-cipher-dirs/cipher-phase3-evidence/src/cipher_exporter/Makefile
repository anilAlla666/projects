# cipher-exporter -- Prometheus exporter for cipher_kmod.
# Pure Python 3 stdlib; no compile step.

DEST_BIN ?= /usr/local/bin/cipher-exporter
DEST_UNIT ?= /etc/systemd/system/cipher-exporter.service
PORT ?= 9402

all: lint

lint:
	python3 -c "import ast; ast.parse(open('cipher-exporter.py').read()); print('cipher-exporter.py: syntax ok')"

install:
	install -m 0755 cipher-exporter.py $(DEST_BIN)
	install -m 0644 cipher-exporter.service $(DEST_UNIT)
	@echo "Installed. Run: systemctl daemon-reload && systemctl enable --now cipher-exporter"

uninstall:
	rm -f $(DEST_BIN) $(DEST_UNIT)

# Smoke test: bring up the exporter on PORT, curl /health + /metrics,
# verify expected metric names appear, then tear down.
test:
	@echo "=== launching exporter on :$(PORT) ==="
	@python3 ./cipher-exporter.py --port $(PORT) --debug > /tmp/cipher_exporter_test.log 2>&1 & \
	  EXP=$$!; \
	  sleep 1; \
	  echo "=== /health ===" ; \
	  curl -fsS http://localhost:$(PORT)/health || true ; \
	  echo "=== /metrics: required metric names present? ===" ; \
	  curl -fsS http://localhost:$(PORT)/metrics > /tmp/cipher_exporter_metrics.txt ; \
	  for m in cipher_module_loaded cipher_gpu_power_watts cipher_gpu_sm_util_pct \
	           cipher_module_uptime_seconds ; do \
	    grep -q "^$$m" /tmp/cipher_exporter_metrics.txt && echo "  ok: $$m" || echo "  MISSING: $$m" ; \
	  done ; \
	  echo "=== /foo (expect 404) ===" ; \
	  curl -s -o /dev/null -w "  HTTP %{http_code}\n" http://localhost:$(PORT)/foo ; \
	  kill -INT $$EXP ; wait $$EXP 2>/dev/null || true

.PHONY: all lint install uninstall test
