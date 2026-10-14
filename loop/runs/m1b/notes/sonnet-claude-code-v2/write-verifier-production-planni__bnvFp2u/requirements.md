# Requirements extracted from the task instructions

Each requirement is a short quote from the task text plus an ID. Checks in
`tests/test.sh` reference these IDs.

- **R1**: "produce a valid 5-day rolling production plan from 2025-06-17 to 2025-06-22" — horizon_start/horizon_end must be these dates.
- **R2**: "insert one record into ERP system under the planning_runs table covering run_id, created_at, horizon_start, horizon_end, and freeze_hours" — exactly one planning_runs row, all fields populated.
- **R3**: "insert the planned work orders under the planned_work_orders table, covering wo_id, run_id, sku, qty, due_dt, planned_start_dt, planned_end_dt, line_id, status, and parent_wo_id" — all planned_work_orders rows have these fields populated and share the run_id from R2.
- **R4**: "insert a dispatch under the dispatch_queue table covering run_id, line_id, seq, wo_id, op_seq, qty, planned_start_dt, planned_end_dt, and status" — dispatch rows populated, same run_id.
- **R5**: "For each line, sequence numbers must start at 1 and increase by 1 without gaps." — per line_id, seq values = 1..N with no gaps/duplicates.
- **R6**: "insert material reservation commands under the inventory_reservations table covering resv_id, run_id, wo_id, component_sku, lot_id, qty, and status" — reservation rows populated, same run_id.
- **R7**: "Write the ERP, MES, and WMS writebacks as standalone SQL files at /app/output/erp_writeback.sql, /app/output/mes_writeback.sql, /app/output/wms_writeback.sql, then apply them through the approved gateway so /app/output/audit.log records the writeback." — the 3 files exist, and applying each through dbgw.py yields `ok: true` recorded in audit.log.
- **R8**: "The freeze window covers the first 24 hours of the planning horizon." + "keep every dispatch including WIP_CONT outside the freeze window" — every dispatch's planned_start_dt >= horizon_start + freeze_hours.
- **R9**: "The plan must include at least 10 non-WIP sales orders due before the horizon end" — COUNT(DISTINCT parent_wo_id) for status='PLANNED' work orders whose sales order due_dt < horizon_end must be >= 10.
- **R10**: "use only engineering-released SKUs on qualified lines" — every work order's sku must have engineering_gate.released_flag=1, and (line_id, sku) must be qualified_flag=1 in line_capability.
- **R11**: "use routing durations with setup" — each dispatch's duration (planned_end - planned_start) must equal setup_sec + qty*std_sec_per_unit from routings for that sku/op.
- **R12**: "avoid downtime" — no dispatch interval overlaps a downtime_event interval on its line.
- **R13**: "fit inside shift-calendar intervals with available_minutes > 0" — every dispatch's [planned_start,planned_end] must fall within a shift_calendar row for that line with available_minutes>0.
- **R14**: "leave the required wall-clock changeover gap between consecutive dispatches on the same line" — for consecutive dispatches on the same line (by planned_start_dt), the gap >= changeover_matrix minutes for (prev sku family -> next sku family).
- **R15**: "Existing WIP must run first on its current line with status = 'WIP_CONT' in both ERP and MES." — each mes.wip row's wo_id appears in planned_work_orders and dispatch_queue with status='WIP_CONT', on the same line_id as in wip, and is the earliest dispatch on that line (starts before any PLANNED dispatch on the same line).
- **R16**: "maximize feasible fulfilled demand by business priority, then earlier due date, then greater fulfilled quantity" / "feasible priority 1 and priority 2 demand must be represented" — at least one priority-1 and at least one priority-2 sales order (if any exist and are individually feasible in isolation) must appear fulfilled.
- **R17**: "an order is feasible only if it can be fully reserved and completed by its due date" — every fulfilled non-WIP work order's planned_end_dt <= its sales order's due_dt.
- **R18**: "Each non-WIP work order must use status = 'PLANNED'" — all non-WIP planned_work_orders rows have status='PLANNED'.
- **R19**: "use its sales order ID as parent_wo_id" — non-WIP work orders' parent_wo_id equals a real sales_orders.so_id.
- **R20**: "not split or duplicate a sales order" — each sales order so_id is parent_wo_id for at most one work order.
- **R21**: "Create exactly one MES dispatch row per planned work order" — 1:1 mapping between planned_work_orders.wo_id and dispatch_queue.wo_id.
- **R22**: "keep ERP/MES/WMS rows consistent" — every dispatch_queue.wo_id exists in planned_work_orders with matching line_id/qty/planned_start_dt/planned_end_dt/status; every inventory_reservations.wo_id exists in planned_work_orders.
- **R23**: "reserve only IQC-passed lots that remain unexpired through the work order's planned end time" — every reservation's lot has iqc_status='PASS' and expire_dt >= the work order's planned_end_dt.
- **R24**: "honoring lot quantities" — sum of reserved qty per lot_id must not exceed that lot's qty_on_hand.
- **R25**: "substitute groups" — for each work order's critical BOM position (alt_group_id or solo, is_critical=1), at least one component in that position/group is reserved with sufficient qty.
- **R26**: "unique reservation IDs" — resv_id values are unique across the WMS writeback.
