-- Only creates the table if it doesn't already exist
-- Prevents errors if the database restarts and tries to run this again
CREATE TABLE IF NOT EXISTS equipment_events (

    -- Auto-incrementing integer ID for each row
    -- SERIAL means PostgreSQL automatically assigns 1, 2, 3, 4...
    id                    SERIAL,

    -- Timestamp of when this row was inserted
    -- TIMESTAMPTZ = timestamp with timezone info included
    -- DEFAULT NOW() = automatically fills in current time if not provided
    time                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- The frame number from the video e.g. 0, 1, 2, 450...
    frame_id              INTEGER,

    -- Our custom readable ID for the machine e.g. "EX-001"
    equipment_id          TEXT,

    -- What type of machine YOLO detected e.g. "truck", "excavator"
    equipment_class       TEXT,

    -- Whether the machine is working: "ACTIVE" or "INACTIVE"
    current_state         TEXT,

    -- What it's doing: "DIGGING", "SWINGING", "DUMPING", "WAITING"
    current_activity      TEXT,

    -- Which part is moving: "arm_only", "full_body", "tracks_only", "none"
    motion_source         TEXT,

    -- Running total of seconds we've been tracking this machine
    total_tracked_seconds FLOAT,

    -- How many of those seconds it was actively working
    total_active_seconds  FLOAT,

    -- How many of those seconds it was idle (tracked - active)
    total_idle_seconds    FLOAT,

    -- Percentage of time it was active e.g. 83.3
    utilization_percent   FLOAT
);

-- Converts the regular PostgreSQL table into a TimescaleDB hypertable
-- A hypertable automatically partitions data into chunks by time
-- This makes queries like "give me last 10 seconds of data" much faster
-- 'time' is the column used for partitioning
SELECT create_hypertable('equipment_events', 'time');