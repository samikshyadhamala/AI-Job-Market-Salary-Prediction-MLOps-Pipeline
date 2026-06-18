CREATE DATABASE IF NOT EXISTS monitoring;

USE monitoring;

CREATE TABLE IF NOT EXISTS drift_metrics (
  id INT AUTO_INCREMENT PRIMARY KEY,
  run_timestamp DATETIME NOT NULL,
  data_quality_score FLOAT,
  feature_drift_share FLOAT,
  prediction_drift_score FLOAT,
  n_drifted_features INT,
  missing_values_count INT,
  schema_errors INT
);

INSERT INTO drift_metrics (run_timestamp, data_quality_score, feature_drift_share, prediction_drift_score, n_drifted_features, missing_values_count, schema_errors)
VALUES
  (NOW() - INTERVAL 60 HOUR, 0.941, 0.167, 0.187, 2, 0, 0),
  (NOW() - INTERVAL 54 HOUR, 0.915, 0.180, 0.190, 2, 1, 0),
  (NOW() - INTERVAL 48 HOUR, 0.902, 0.175, 0.185, 2, 1, 0),
  (NOW() - INTERVAL 42 HOUR, 0.888, 0.165, 0.180, 2, 0, 0),
  (NOW() - INTERVAL 36 HOUR, 0.882, 0.160, 0.175, 2, 0, 0),
  (NOW() - INTERVAL 30 HOUR, 0.895, 0.148, 0.160, 2, 0, 0),
  (NOW() - INTERVAL 24 HOUR, 0.911, 0.152, 0.155, 2, 0, 0),
  (NOW() - INTERVAL 18 HOUR, 0.920, 0.145, 0.150, 2, 0, 0),
  (NOW() - INTERVAL 12 HOUR, 0.934, 0.135, 0.142, 2, 0, 0),
  (NOW() - INTERVAL 6 HOUR, 0.941, 0.130, 0.135, 2, 0, 0),
  (NOW() - INTERVAL 2 HOUR, 0.985, 0.075, 0.085, 1, 0, 0),
  (NOW(), 0.985, 0.075, 0.085, 1, 0, 0);
