# Requirements Extracted from Task Instructions

R1. Use the gateway tool for all database interaction: "together with the provided gateway tool `/app/data/dbgw.py` to inspect the available databases" and "apply them through the approved gateway so `/app/output/audit.log` records the writeback" - the three SQL files must be applied through dbgw.py (not raw sqlite), and this must be reflected as successful (ok:true) SQL_EXEC entries in `/app/output/audit.log` for all three systems (erp, mes, wms).

R2. "produce a valid 5-day rolling production plan from 2025-06-17 to 2025-06-22" - the planning_runs row's horizon_start must be 2025-06-17T00:00:00Z and horizon_end must be 2025-06-22T00:00:00Z (a 5-day span), with the same run_id used consistently across ERP/MES/WMS writebacks.

R3. "insert one record into ERP system under the planning_runs table covering run_id, created_at, horizon_start, horizon_end, and freeze_hours" - exactly one planning_runs INSERT with all these columns populated (non-null).

R4. "insert the planned work orders under the planned_work_orders table, covering wo_id, run_id, sku, qty, due_dt, planned_start_dt, planned_end_dt, line_id, status, and parent_wo_id" - all planned_work_orders inserts must supply all these columns.

R5. "insert a dispatch under the dispatch_queue table covering run_id, line_id, seq, wo_id, op_seq, qty, planned_start_dt, planned_end_dt, and status" - all dispatch_queue inserts must supply all these columns.

R6. "For each line, sequence numbers must start at 1 and increase by 1 without gaps." - per line_id, the set of `seq` values in dispatch_queue must be exactly {1,2,...,N} with no gaps or duplicates.

R7. "insert material reservation commands under the inventory_reservations table covering resv_id, run_id, wo_id, component_sku, lot_id, qty, and status" - all inventory_reservations inserts must supply all these columns.

R8. "Write the ERP, MES, and WMS writebacks as standalone SQL files at `/app/output/erp_writeback.sql`, `/app/output/mes_writeback.sql`, and `/app/output/wms_writeback.sql`" - these three files must exist and each must only touch its respective system's allowed table(s) (erp_writeback.sql -> planning_runs/planned_work_orders; mes_writeback.sql -> dispatch_queue; wms_writeback.sql -> inventory_reservations), consistent with the gateway's role-based policy (erp_planner, mes_scheduler, wms_materials).

R9. "The freeze window covers the first 24 hours of the planning horizon." - planning_runs.freeze_hours must be 24, and no non-WIP (status != 'WIP_CONT') dispatch/work-order may have planned_start_dt earlier than horizon_start + 24h.

R10. "The plan must include at least 10 non-WIP sales orders due before the horizon end" - the plan must reference at least 10 distinct non-WIP (status='PLANNED') work orders whose parent_wo_id corresponds to a sales_orders due_dt < horizon_end.

R11. "use only engineering-released SKUs on qualified lines" - every planned_work_orders/dispatch_queue sku must have engineering_gate.released_flag=1, and its line_id must have line_capability.qualified_flag=1 for that sku.

R12. "use routing durations with setup" - for each non-WIP work order, planned_end_dt - planned_start_dt must equal sum over routing rows for that sku of (setup_sec + std_sec_per_unit*qty) within reasonable tolerance.

R13. "keep every dispatch including WIP_CONT outside the freeze window" - every dispatch_queue planned_start_dt (including WIP_CONT rows) must be >= horizon_start + freeze_hours.

R14. "avoid downtime" - no dispatch's [planned_start_dt, planned_end_dt) interval may overlap a downtime_event interval for that line_id.

R15. "fit inside shift-calendar intervals with `available_minutes > 0`" - every dispatch's [planned_start_dt, planned_end_dt) must be fully contained within some shift_calendar interval for that line with available_minutes>0.

R16. "leave the required wall-clock changeover gap between consecutive dispatches on the same line" - for consecutive dispatches (by seq) on the same line, the gap between the end of one and the start of the next must be >= changeover_matrix minutes for (from_family,to_family) of the respective SKUs' program_family.

R17. "Existing WIP must run first on its current line with status = 'WIP_CONT' in both ERP and MES" - for each wip table row, there must be a planned_work_orders row and a dispatch_queue row with matching wo_id, status='WIP_CONT', on the same line_id, and it must be the first (lowest seq) dispatch on that line.

R18. "an order is feasible only if it can be fully reserved and completed by its due date" / "maximize feasible fulfilled demand by business priority, then earlier due date, then greater fulfilled quantity" - for every non-WIP planned_work_orders row, planned_end_dt <= due_dt (the linked sales order's due_dt).

R19. "feasible priority 1 and priority 2 demand must be represented" - if any priority-1 sales order is feasible (capacity+inventory workable in isolation), at least one priority-1 order must appear among planned work orders; likewise for priority-2.

R20. "Each non-WIP work order must use status = 'PLANNED'" - every planned_work_orders row with a non-null parent_wo_id must have status='PLANNED'.

R21. "use its sales order ID as `parent_wo_id`" - every non-WIP planned_work_orders.parent_wo_id must equal a real sales_orders.so_id.

R22. "not split or duplicate a sales order" - each sales_orders.so_id must be used as parent_wo_id by at most one planned_work_orders row.

R23. "Create exactly one MES dispatch row per planned work order" - for every planned_work_orders.wo_id, there must be exactly one dispatch_queue row with that wo_id.

R24. "keep ERP/MES/WMS rows consistent" - every dispatch_queue.wo_id must exist in planned_work_orders; every inventory_reservations.wo_id must exist in planned_work_orders; run_id must match across all tables.

R25. "reserve only IQC-passed lots that remain unexpired through the work order's planned end time" - every inventory_reservations row must reference an inventory_lots row with iqc_status='PASS' and expire_dt >= the referencing work order's planned_end_dt.

R26. "honoring lot quantities" - for each (component_sku, lot_id), the sum of qty reserved across all inventory_reservations rows must not exceed that lot's qty_on_hand.

R27. "unique reservation IDs" - all resv_id values in inventory_reservations must be unique.
