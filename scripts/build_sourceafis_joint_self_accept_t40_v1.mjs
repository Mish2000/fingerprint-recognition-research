import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const ROOT = path.resolve(path.dirname(SCRIPT_PATH), "..");
const COHORT_NAME = "sourceafis_joint_self_accept_t40_v1";
const COHORT_VERSION = "v1";
const COHORT_DESCRIPTION = "SourceAFIS-selected joint self-accepted cohort at threshold 40";
const THRESHOLD = 40.0;
const BUNDLE_ID = "cb6ed29d0231c44a3d95e60fd1b9fd7aa8f2fa333c8ccecd971b252c041830c3";
const OUTPUT_ROOT = path.join(ROOT, "results", "cohorts", COHORT_NAME);
const THRESHOLD_AUDIT_PATH = path.join(
  ROOT,
  "results",
  "sourceafis",
  "pairwise-benchmark-v2",
  "threshold40_audit",
  "threshold40_audit.json",
);
const CONDITIONS = [
  ["sd300b", "plain_self"],
  ["sd300b", "roll_self"],
  ["sd300b", "plain_roll"],
  ["sd300c", "plain_self"],
  ["sd300c", "roll_self"],
  ["sd300c", "plain_roll"],
];
const SELF_KEYS = [
  "sd300b/plain_self",
  "sd300b/roll_self",
  "sd300c/plain_self",
  "sd300c/roll_self",
];
const REASONS = [
  "sd300b_plain_self_below_40",
  "sd300b_roll_self_below_40",
  "sd300c_plain_self_below_40",
  "sd300c_roll_self_below_40",
  "missing_required_protocol_identity",
];
const SELF_REASON_BY_KEY = {
  "sd300b/plain_self": "sd300b_plain_self_below_40",
  "sd300b/roll_self": "sd300b_roll_self_below_40",
  "sd300c/plain_self": "sd300c_plain_self_below_40",
  "sd300c/roll_self": "sd300c_roll_self_below_40",
};
const RESULT_COLUMNS = [
  "pair_id", "dataset", "protocol", "subject_id", "canonical_finger_position",
  "method", "method_version", "benchmark_contract_version", "result_schema_version",
  "config_hash", "implementation_hash", "manifest_sha256", "score_direction",
  "score_semantics", "raw_score", "prepare_a_ms", "prepare_b_ms", "compare_ms",
  "method_prepare_a_ms", "method_prepare_b_ms", "method_compare_ms", "total_ms",
  "prepare_a_diagnostics", "prepare_b_diagnostics", "compare_diagnostics", "status",
  "error_code", "error_message",
];
const MANIFEST_COLUMNS = [
  "pair_id", "dataset", "protocol", "subject_id", "canonical_finger_position",
  "ppi", "raw_frgp_a", "raw_frgp_b", "path_a", "path_b",
];
const COHORT_RULE = [
  "The identity unit is (subject_id, canonical_finger_position).",
  "The base population is the exact shared identity set in the sd300b/plain_roll and sd300c/plain_roll primary SourceAFIS pairwise-benchmark-v2 results.",
  "An identity is included only when it exists in all six required protocols and has raw_score >= 40 in sd300b/plain_self, sd300b/roll_self, sd300c/plain_self, and sd300c/roll_self.",
  "The sd300b/plain_roll and sd300c/plain_roll scores, decisions, and statuses are never used for inclusion or exclusion.",
].join(" ");

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function rel(filePath) {
  return path.relative(ROOT, filePath).split(path.sep).join("/");
}

function conditionKey(dataset, protocol) {
  return `${dataset}/${protocol}`;
}

function identityKey(subjectId, fingerPosition) {
  return `${subjectId}\u001f${Number(fingerPosition)}`;
}

function compareIdentity(left, right) {
  return left.subject_id.localeCompare(right.subject_id, "en")
    || Number(left.canonical_finger_position) - Number(right.canonical_finger_position);
}

function arraysEqual(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function rounded(value) {
  return Number(value.toFixed(12));
}

function numberText(value) {
  return Number(value.toFixed(12)).toString();
}

function percent(count, total) {
  return total === 0 ? 0 : rounded((count / total) * 100);
}

function mean(values) {
  return values.length === 0 ? null : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function percentileNearestRank(values, fraction) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.max(0, Math.ceil(fraction * sorted.length) - 1)];
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function csvText(headers, rows) {
  return `${[headers, ...rows].map((row) => row.map(csvEscape).join(",")).join("\n")}\n`;
}

function scanCsvRecords(text) {
  const records = [];
  let start = 0;
  let inQuotes = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (char === '"') {
      if (inQuotes && text[index + 1] === '"') index += 1;
      else inQuotes = !inQuotes;
    } else if (!inQuotes && (char === "\n" || char === "\r")) {
      records.push(text.slice(start, index));
      if (char === "\r" && text[index + 1] === "\n") index += 1;
      start = index + 1;
    }
  }
  assert(!inQuotes, "Unterminated quoted CSV field");
  if (start < text.length) records.push(text.slice(start));
  while (records.length > 0 && records.at(-1) === "") records.pop();
  return records;
}

function parseCsvRecord(record) {
  const fields = [];
  let value = "";
  let inQuotes = false;
  for (let index = 0; index < record.length; index += 1) {
    const char = record[index];
    if (inQuotes) {
      if (char === '"') {
        if (record[index + 1] === '"') {
          value += '"';
          index += 1;
        } else inQuotes = false;
      } else value += char;
    } else if (char === ",") {
      fields.push(value);
      value = "";
    } else if (char === '"' && value === "") inQuotes = true;
    else value += char;
  }
  assert(!inQuotes, "Unterminated quoted CSV field in record");
  fields.push(value);
  return fields;
}

async function readCsv(filePath) {
  const text = await fs.readFile(filePath, "utf8");
  const records = scanCsvRecords(text);
  assert(records.length >= 1, `Empty CSV: ${rel(filePath)}`);
  const headers = parseCsvRecord(records[0]);
  assert(new Set(headers).size === headers.length, `Duplicate CSV header in ${rel(filePath)}`);
  const rows = records.slice(1).map((rawRecord, rowIndex) => {
    const fields = parseCsvRecord(rawRecord);
    assert(fields.length === headers.length, `Column mismatch at row ${rowIndex + 2} in ${rel(filePath)}`);
    return { ...Object.fromEntries(headers.map((header, index) => [header, fields[index]])), rawRecord };
  });
  const eol = text.includes("\r\n") ? "\r\n" : "\n";
  return { filePath, text, records, headers, headerRecord: records[0], rows, eol };
}

async function sha256(filePath) {
  const data = await fs.readFile(filePath);
  return crypto.createHash("sha256").update(data).digest("hex");
}

async function collectFiles(directory) {
  const files = [];
  for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
    const candidate = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await collectFiles(candidate));
    else if (entry.isFile()) files.push(candidate);
  }
  return files.sort((left, right) => rel(left).localeCompare(rel(right), "en"));
}

async function hashFiles(filePaths) {
  return Object.fromEntries(await Promise.all(filePaths.map(async (filePath) => [rel(filePath), await sha256(filePath)])));
}

async function writeAtomic(filePath, content) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.tmp`;
  await fs.writeFile(temporary, content, "utf8");
  await fs.rename(temporary, filePath);
}

function resultPath(dataset, protocol) {
  return path.join(
    ROOT,
    "results",
    dataset,
    protocol,
    "sourceafis",
    "pairwise-benchmark-v2",
    BUNDLE_ID,
    "pairs.csv",
  );
}

function manifestPath(dataset, protocol) {
  return path.join(ROOT, "protocols", dataset, `${protocol}.csv`);
}

function projectionPath(dataset, protocol) {
  return path.join(OUTPUT_ROOT, dataset, protocol, "pairs.csv");
}

function finiteNumber(raw, label) {
  assert(raw !== "", `Missing ${label}`);
  const value = Number(raw);
  assert(Number.isFinite(value), `Non-finite ${label}: ${raw}`);
  return value;
}

async function readAndValidateSource(dataset, protocol, manifestHash) {
  const key = conditionKey(dataset, protocol);
  const [table, manifest] = await Promise.all([
    readCsv(resultPath(dataset, protocol)),
    readCsv(manifestPath(dataset, protocol)),
  ]);
  assert(arraysEqual(table.headers, RESULT_COLUMNS), `Result schema mismatch in ${key}`);
  assert(arraysEqual(manifest.headers, MANIFEST_COLUMNS), `Manifest schema mismatch in ${key}`);
  assert(table.rows.length === manifest.rows.length, `Result/manifest row count mismatch in ${key}`);
  const pairIds = new Set();
  const identities = new Set();
  for (let index = 0; index < table.rows.length; index += 1) {
    const row = table.rows[index];
    const manifestRow = manifest.rows[index];
    for (const field of ["pair_id", "dataset", "protocol", "subject_id", "canonical_finger_position"]) {
      assert(row[field] === manifestRow[field], `${field} differs from manifest at row ${index + 2} in ${key}`);
    }
    assert(row.dataset === dataset && row.protocol === protocol, `Wrong condition in ${key}`);
    assert(row.manifest_sha256 === manifestHash, `Manifest SHA field mismatch for ${row.pair_id}`);
    assert(!pairIds.has(row.pair_id), `Duplicate pair_id in ${key}: ${row.pair_id}`);
    pairIds.add(row.pair_id);
    const identity = identityKey(row.subject_id, row.canonical_finger_position);
    assert(!identities.has(identity), `Duplicate identity in ${key}: ${identity}`);
    identities.add(identity);
    if (row.raw_score !== "") finiteNumber(row.raw_score, `raw_score for ${row.pair_id}`);
    for (const timing of [
      "prepare_a_ms", "prepare_b_ms", "compare_ms", "method_prepare_a_ms",
      "method_prepare_b_ms", "method_compare_ms", "total_ms",
    ]) {
      if (row[timing] !== "") {
        const value = finiteNumber(row[timing], `${timing} for ${row.pair_id}`);
        assert(value >= 0, `Negative ${timing} for ${row.pair_id}`);
      }
    }
    row.key = identity;
    row.score = row.raw_score === "" ? null : Number(row.raw_score);
  }
  table.rowMap = new Map(table.rows.map((row) => [row.key, row]));
  table.pairMap = new Map(table.rows.map((row) => [row.pair_id, row]));
  table.identities = identities;
  return { table, manifest };
}

function summarizeProjection(dataset, protocol, table) {
  const okRows = table.rows.filter((row) => row.status === "ok");
  const nonOkCount = table.rows.length - okRows.length;
  const scores = okRows.map((row) => finiteNumber(row.raw_score, `projection raw_score for ${row.pair_id}`));
  const methodCompare = table.rows
    .filter((row) => row.method_compare_ms !== "")
    .map((row) => finiteNumber(row.method_compare_ms, `projection method_compare_ms for ${row.pair_id}`));
  const scoreZeroCount = scores.filter((score) => score === 0).length;
  const positiveBelowCount = scores.filter((score) => score > 0 && score < THRESHOLD).length;
  const acceptedCount = scores.filter((score) => score >= THRESHOLD).length;
  const rejectedCount = scores.filter((score) => score < THRESHOLD).length;
  assert(acceptedCount + rejectedCount + nonOkCount === table.rows.length, `Decision counts do not reconcile in ${dataset}/${protocol}`);
  assert(scoreZeroCount + positiveBelowCount === rejectedCount, `Rejected buckets do not reconcile in ${dataset}/${protocol}`);
  const subjects = new Set(table.rows.map((row) => row.subject_id));
  const fingerPositions = Object.fromEntries(Array.from({ length: 10 }, (_, index) => index + 1).map((position) => [
    String(position), table.rows.filter((row) => Number(row.canonical_finger_position) === position).length,
  ]));
  const groupedFingers = {
    thumb: fingerPositions["1"] + fingerPositions["6"],
    index: fingerPositions["2"] + fingerPositions["7"],
    middle: fingerPositions["3"] + fingerPositions["8"],
    ring: fingerPositions["4"] + fingerPositions["9"],
    little: fingerPositions["5"] + fingerPositions["10"],
  };
  return {
    dataset,
    protocol,
    distinct_subject_count: subjects.size,
    identity_count: table.rows.length,
    canonical_finger_distribution: fingerPositions,
    canonical_finger_group_counts: groupedFingers,
    method_compare_observation_count: methodCompare.length,
    mean_method_compare_ms: rounded(mean(methodCompare)),
    median_method_compare_ms: rounded(median(methodCompare)),
    p95_method_compare_ms: rounded(percentileNearestRank(methodCompare, 0.95)),
    ok_count: okRows.length,
    non_ok_count: nonOkCount,
    score_zero_count: scoreZeroCount,
    score_positive_below_40_count: positiveBelowCount,
    accepted_at_threshold_40_count: acceptedCount,
    rejected_at_threshold_40_count: rejectedCount,
    accepted_percentage: percent(acceptedCount, table.rows.length),
    rejected_percentage: percent(rejectedCount, table.rows.length),
    mean_score: rounded(mean(scores)),
    median_score: rounded(median(scores)),
  };
}

function markdownNumber(value) {
  return Number(value.toFixed(6)).toString();
}

function buildSupervisorTables(summaries) {
  const lines = [
    "# SourceAFIS supervisor tables — joint self-accepted cohort at threshold 40",
    "",
    `Cohort: \`${COHORT_NAME}\` — ${COHORT_DESCRIPTION}.`,
    "",
    "This is a SourceAFIS-selected cohort. It does not replace the complete, unfiltered results for scientific comparison between methods.",
    "",
    "The identity unit is `(subject_id, canonical_finger_position)`. Thumb combines canonical positions 1 and 6; index 2 and 7; middle 3 and 8; ring 4 and 9; little 5 and 10.",
    "",
  ];
  for (const summary of summaries) {
    lines.push(
      `## ${summary.dataset}/${summary.protocol}`,
      "",
      "| distinct subjects | identities | thumb | index | middle | ring | little | mean method_compare_ms | median method_compare_ms | score=0 | 0<score<40 | accepted >=40 | rejected <40 | accepted percentage |",
      "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
      `| ${summary.distinct_subject_count} | ${summary.identity_count} | ${summary.canonical_finger_group_counts.thumb} | ${summary.canonical_finger_group_counts.index} | ${summary.canonical_finger_group_counts.middle} | ${summary.canonical_finger_group_counts.ring} | ${summary.canonical_finger_group_counts.little} | ${markdownNumber(summary.mean_method_compare_ms)} | ${markdownNumber(summary.median_method_compare_ms)} | ${summary.score_zero_count} | ${summary.score_positive_below_40_count} | ${summary.accepted_at_threshold_40_count} | ${summary.rejected_at_threshold_40_count} | ${markdownNumber(summary.accepted_percentage)}% |`,
      "",
    );
  }
  return `${lines.join("\n")}\n`;
}

async function main() {
  assert(path.resolve(OUTPUT_ROOT).startsWith(path.resolve(ROOT) + path.sep), "Unsafe output root");

  const sourceafisFiles = [];
  for (const [dataset, protocol] of CONDITIONS) {
    sourceafisFiles.push(...await collectFiles(path.join(ROOT, "results", dataset, protocol, "sourceafis")));
  }
  const manifestFiles = CONDITIONS.map(([dataset, protocol]) => manifestPath(dataset, protocol));
  const protectedFiles = [
    ...sourceafisFiles,
    ...manifestFiles,
    ...await collectFiles(path.join(ROOT, "results", "sourceafis", "pairwise-benchmark-v2", "zero_score_audit")),
    ...await collectFiles(path.join(ROOT, "results", "sourceafis", "pairwise-benchmark-v2", "threshold40_audit")),
    ...await collectFiles(path.join(ROOT, "results", "cohorts", "sourceafis_joint_self_positive_v1")),
  ].sort((left, right) => rel(left).localeCompare(rel(right), "en"));
  const uniqueProtectedFiles = [...new Map(protectedFiles.map((filePath) => [path.resolve(filePath), filePath])).values()];
  const beforeHashes = await hashFiles(uniqueProtectedFiles);

  const thresholdAudit = JSON.parse(await fs.readFile(THRESHOLD_AUDIT_PATH, "utf8"));
  assert(thresholdAudit.sourceafis_threshold === THRESHOLD, "Threshold audit does not use threshold 40");
  assert(thresholdAudit.validation.protected_source_artifacts_unchanged_during_audit === true, "Threshold audit source validation is not affirmative");

  const sourceResultHashes = Object.fromEntries(await Promise.all(CONDITIONS.map(async ([dataset, protocol]) => [
    rel(resultPath(dataset, protocol)), await sha256(resultPath(dataset, protocol)),
  ])));
  const sourceManifestHashes = Object.fromEntries(await Promise.all(CONDITIONS.map(async ([dataset, protocol]) => [
    rel(manifestPath(dataset, protocol)), await sha256(manifestPath(dataset, protocol)),
  ])));

  const sources = new Map();
  for (const [dataset, protocol] of CONDITIONS) {
    const key = conditionKey(dataset, protocol);
    sources.set(key, await readAndValidateSource(dataset, protocol, sourceManifestHashes[rel(manifestPath(dataset, protocol))]));
  }

  const baseB = [...sources.get("sd300b/plain_roll").table.identities].sort();
  const baseC = [...sources.get("sd300c/plain_roll").table.identities].sort();
  assert(arraysEqual(baseB, baseC), "sd300b/plain_roll and sd300c/plain_roll identity sets differ");
  const auditPlainRollTotals = thresholdAudit.threshold_summaries
    .filter((row) => row.scope === "primary_full_results" && row.protocol === "plain_roll")
    .map((row) => row.total_count);
  assert(auditPlainRollTotals.length === 2 && auditPlainRollTotals.every((count) => count === baseB.length), "Base identity count does not match threshold audit");

  const identities = baseB.map((key) => {
    const reference = sources.get("sd300b/plain_roll").table.rowMap.get(key);
    const missingRequired = CONDITIONS.some(([dataset, protocol]) => !sources.get(conditionKey(dataset, protocol)).table.rowMap.has(key));
    const selfScores = {};
    const reasons = [];
    for (const selfKey of SELF_KEYS) {
      const row = sources.get(selfKey).table.rowMap.get(key);
      if (!row) {
        selfScores[selfKey] = null;
        continue;
      }
      assert(row.status === "ok", `Non-ok self result cannot be classified for ${key} in ${selfKey}`);
      const score = finiteNumber(row.raw_score, `self raw_score for ${key} in ${selfKey}`);
      selfScores[selfKey] = score;
      if (score < THRESHOLD) reasons.push(SELF_REASON_BY_KEY[selfKey]);
    }
    if (missingRequired) reasons.push("missing_required_protocol_identity");
    const included = !missingRequired && SELF_KEYS.every((selfKey) => selfScores[selfKey] >= THRESHOLD);
    assert(included === (reasons.length === 0), `Reason coverage failure for ${key}`);
    return {
      key,
      subject_id: reference.subject_id,
      canonical_finger_position: Number(reference.canonical_finger_position),
      selfScores,
      reasons,
      included,
    };
  }).sort(compareIdentity);

  const included = identities.filter((identity) => identity.included);
  const excluded = identities.filter((identity) => !identity.included);
  const includedKeys = new Set(included.map((identity) => identity.key));
  const includedSubjects = new Set(included.map((identity) => identity.subject_id));
  const auditExpected = thresholdAudit.self_protocol_audit;
  assert(included.length === auditExpected.possible_threshold40_cohort_identity_count, "Included identity count differs from threshold audit");
  assert(includedSubjects.size === auditExpected.possible_threshold40_cohort_distinct_subject_count, "Included subject count differs from threshold audit");
  assert(excluded.length === baseB.length - auditExpected.possible_threshold40_cohort_identity_count, "Excluded identity count differs from threshold audit-derived expectation");

  const scoreColumns = CONDITIONS.map(([dataset, protocol]) => `${dataset}_${protocol}_raw_score`);
  const includedHeaders = ["subject_id", "canonical_finger_position", ...scoreColumns];
  const includedPath = path.join(OUTPUT_ROOT, "included_identities.csv");
  await writeAtomic(includedPath, csvText(includedHeaders, included.map((identity) => [
    identity.subject_id,
    identity.canonical_finger_position,
    ...CONDITIONS.map(([dataset, protocol]) => sources.get(conditionKey(dataset, protocol)).table.rowMap.get(identity.key).raw_score),
  ])));

  const excludedHeaders = [
    "subject_id",
    "canonical_finger_position",
    "reason_flags",
    ...REASONS,
    ...SELF_KEYS.map((key) => `${key.replace("/", "_")}_raw_score`),
  ];
  const excludedPath = path.join(OUTPUT_ROOT, "excluded_identities.csv");
  await writeAtomic(excludedPath, csvText(excludedHeaders, excluded.map((identity) => [
    identity.subject_id,
    identity.canonical_finger_position,
    identity.reasons.join(";"),
    ...REASONS.map((reason) => identity.reasons.includes(reason)),
    ...SELF_KEYS.map((key) => identity.selfScores[key] ?? ""),
  ])));

  const projectionTables = new Map();
  const projectionHashes = {};
  for (const [dataset, protocol] of CONDITIONS) {
    const key = conditionKey(dataset, protocol);
    const source = sources.get(key).table;
    const rows = source.rows.filter((row) => includedKeys.has(row.key));
    assert(rows.length === included.length, `Projection row count mismatch in ${key}`);
    assert(arraysEqual(rows.map((row) => row.key).sort(), [...includedKeys].sort()), `Projection identity set mismatch in ${key}`);
    assert(new Set(rows.map((row) => row.pair_id)).size === rows.length, `Projection pair_id duplicate in ${key}`);
    const projection = projectionPath(dataset, protocol);
    const text = `${source.headerRecord}${source.eol}${rows.map((row) => row.rawRecord).join(source.eol)}${source.eol}`;
    await writeAtomic(projection, text);
    const written = await readCsv(projection);
    assert(written.headerRecord === source.headerRecord, `Projection header changed in ${key}`);
    assert(arraysEqual(written.rows.map((row) => row.rawRecord), rows.map((row) => row.rawRecord)), `Projection rows changed in ${key}`);
    written.rows = written.rows.map((row) => ({
      ...row,
      key: identityKey(row.subject_id, row.canonical_finger_position),
      score: row.raw_score === "" ? null : Number(row.raw_score),
    }));
    written.rowMap = new Map(written.rows.map((row) => [row.key, row]));
    projectionTables.set(key, written);
    projectionHashes[rel(projection)] = await sha256(projection);
  }

  const commonProjectionKeys = [...includedKeys].sort();
  for (const [key, table] of projectionTables) {
    assert(arraysEqual([...table.rowMap.keys()].sort(), commonProjectionKeys), `Projection identity equality failed in ${key}`);
  }
  for (const selfKey of SELF_KEYS) {
    const table = projectionTables.get(selfKey);
    assert(table.rows.every((row) => row.status === "ok" && Number(row.raw_score) >= THRESHOLD), `Self projection contains rejection in ${selfKey}`);
  }

  const summaries = CONDITIONS.map(([dataset, protocol]) => summarizeProjection(dataset, protocol, projectionTables.get(conditionKey(dataset, protocol))));
  for (const summary of summaries.filter((item) => item.protocol !== "plain_roll")) {
    assert(summary.rejected_at_threshold_40_count === 0, `Self rejected count is not zero in ${summary.dataset}/${summary.protocol}`);
    assert(summary.accepted_percentage === 100, `Self accepted percentage is not 100 in ${summary.dataset}/${summary.protocol}`);
  }

  const includedFingerDistribution = Object.fromEntries(Array.from({ length: 10 }, (_, index) => index + 1).map((position) => [
    String(position), included.filter((identity) => identity.canonical_finger_position === position).length,
  ]));
  assert(JSON.stringify(includedFingerDistribution) === JSON.stringify(auditExpected.possible_threshold40_cohort_canonical_finger_distribution), "Finger distribution differs from threshold audit");

  const exclusionCounts = Object.fromEntries(REASONS.map((reason) => [
    reason, excluded.filter((identity) => identity.reasons.includes(reason)).length,
  ]));
  const exactCombinationCounts = {};
  for (const identity of excluded) {
    const combination = identity.reasons.join(";");
    exactCombinationCounts[combination] = (exactCombinationCounts[combination] ?? 0) + 1;
  }
  const orderedExactCombinationCounts = Object.fromEntries(Object.entries(exactCombinationCounts).sort(([left], [right]) => left.localeCompare(right, "en")));
  const pairwiseIntersections = {};
  for (let left = 0; left < REASONS.length; left += 1) {
    for (let right = left + 1; right < REASONS.length; right += 1) {
      const key = `${REASONS[left]} & ${REASONS[right]}`;
      pairwiseIntersections[key] = excluded.filter((identity) => identity.reasons.includes(REASONS[left]) && identity.reasons.includes(REASONS[right])).length;
    }
  }
  const multipleReasonCount = excluded.filter((identity) => identity.reasons.length > 1).length;
  assert(excluded.every((identity) => identity.reasons.some((reason) => REASONS.includes(reason))), "Excluded identity lacks allowed reason");

  const supervisorPath = path.join(OUTPUT_ROOT, "supervisor_tables.md");
  await writeAtomic(supervisorPath, buildSupervisorTables(summaries));

  const includedHash = await sha256(includedPath);
  const excludedHash = await sha256(excludedPath);
  const supervisorHash = await sha256(supervisorPath);
  const thresholdAuditHash = await sha256(THRESHOLD_AUDIT_PATH);
  const summaryHeaders = [
    "cohort_name", "cohort_description", "threshold", "dataset", "protocol",
    "base_identity_count", "included_identity_count", "excluded_identity_count",
    "distinct_subject_count", "identity_count", "thumb", "index", "middle", "ring", "little",
    "mean_method_compare_ms", "median_method_compare_ms", "p95_method_compare_ms",
    "score_zero_count", "score_positive_below_40_count", "accepted_at_threshold_40_count",
    "rejected_at_threshold_40_count", "non_ok_count", "accepted_percentage", "rejected_percentage",
    "mean_score", "median_score", "source_result_path", "source_result_sha256",
    "manifest_path", "manifest_sha256", "projection_path", "projection_sha256",
    "included_identities_sha256", "excluded_identities_sha256", "supervisor_tables_sha256",
    "threshold40_audit_sha256",
  ];
  const summaryCsvPath = path.join(OUTPUT_ROOT, "cohort_summary.csv");
  await writeAtomic(summaryCsvPath, csvText(summaryHeaders, summaries.map((summary) => {
    const source = resultPath(summary.dataset, summary.protocol);
    const manifest = manifestPath(summary.dataset, summary.protocol);
    const projection = projectionPath(summary.dataset, summary.protocol);
    const values = {
      cohort_name: COHORT_NAME,
      cohort_description: COHORT_DESCRIPTION,
      threshold: THRESHOLD,
      dataset: summary.dataset,
      protocol: summary.protocol,
      base_identity_count: identities.length,
      included_identity_count: included.length,
      excluded_identity_count: excluded.length,
      distinct_subject_count: summary.distinct_subject_count,
      identity_count: summary.identity_count,
      ...summary.canonical_finger_group_counts,
      mean_method_compare_ms: numberText(summary.mean_method_compare_ms),
      median_method_compare_ms: numberText(summary.median_method_compare_ms),
      p95_method_compare_ms: numberText(summary.p95_method_compare_ms),
      score_zero_count: summary.score_zero_count,
      score_positive_below_40_count: summary.score_positive_below_40_count,
      accepted_at_threshold_40_count: summary.accepted_at_threshold_40_count,
      rejected_at_threshold_40_count: summary.rejected_at_threshold_40_count,
      non_ok_count: summary.non_ok_count,
      accepted_percentage: numberText(summary.accepted_percentage),
      rejected_percentage: numberText(summary.rejected_percentage),
      mean_score: numberText(summary.mean_score),
      median_score: numberText(summary.median_score),
      source_result_path: rel(source),
      source_result_sha256: sourceResultHashes[rel(source)],
      manifest_path: rel(manifest),
      manifest_sha256: sourceManifestHashes[rel(manifest)],
      projection_path: rel(projection),
      projection_sha256: projectionHashes[rel(projection)],
      included_identities_sha256: includedHash,
      excluded_identities_sha256: excludedHash,
      supervisor_tables_sha256: supervisorHash,
      threshold40_audit_sha256: thresholdAuditHash,
    };
    return summaryHeaders.map((header) => values[header]);
  })));

  const summaryCsvHash = await sha256(summaryCsvPath);
  const plainRollSummaries = summaries.filter((summary) => summary.protocol === "plain_roll").map((summary) => ({
    dataset: summary.dataset,
    protocol: summary.protocol,
    total_genuine_pairs: summary.identity_count,
    genuine_accept_count: summary.accepted_at_threshold_40_count,
    false_non_match_count: summary.rejected_at_threshold_40_count,
    score_zero_count: summary.score_zero_count,
    score_positive_below_40_count: summary.score_positive_below_40_count,
    non_ok_count: summary.non_ok_count,
    genuine_accept_percentage: summary.accepted_percentage,
    false_non_match_percentage: summary.rejected_percentage,
    mean_score: summary.mean_score,
    median_score: summary.median_score,
    mean_method_compare_ms: summary.mean_method_compare_ms,
    median_method_compare_ms: summary.median_method_compare_ms,
    p95_method_compare_ms: summary.p95_method_compare_ms,
  }));

  const afterHashes = await hashFiles(uniqueProtectedFiles);
  assert(JSON.stringify(beforeHashes) === JSON.stringify(afterHashes), "A protected source artifact changed during cohort construction");

  const summary = {
    cohort_name: COHORT_NAME,
    cohort_version: COHORT_VERSION,
    cohort_description: COHORT_DESCRIPTION,
    selection_note: "This is a SourceAFIS-selected cohort and does not replace the complete, unfiltered results for scientific comparison between methods.",
    identity_unit: ["subject_id", "canonical_finger_position"],
    threshold: THRESHOLD,
    threshold_source: {
      path: rel(THRESHOLD_AUDIT_PATH),
      sha256: thresholdAuditHash,
      description: "SOURCEAFIS_THRESHOLD = 40.0; self-accepted means status == ok and raw_score >= 40.",
    },
    inclusion_rule: COHORT_RULE,
    plain_roll_inclusion_dependency: "none",
    deterministic_outputs: true,
    timestamps_in_deterministic_outputs: false,
    summary_numeric_rounding_decimal_places: 12,
    percentile_method: "nearest-rank: sorted_values[ceil(0.95 * n) - 1]",
    counts: {
      base_identity_count: identities.length,
      included_identity_count: included.length,
      excluded_identity_count: excluded.length,
      included_subject_count: includedSubjects.size,
      included_canonical_finger_distribution: includedFingerDistribution,
      exclusion_count_by_reason_flag: exclusionCounts,
      excluded_identity_count_with_multiple_reasons: multipleReasonCount,
      exclusion_count_by_exact_reason_combination: orderedExactCombinationCounts,
      exclusion_pairwise_intersection_counts: pairwiseIntersections,
    },
    condition_summaries: summaries,
    plain_roll_summaries: plainRollSummaries,
    provenance: {
      builder_path: rel(SCRIPT_PATH),
      builder_sha256: await sha256(SCRIPT_PATH),
      threshold40_audit_path: rel(THRESHOLD_AUDIT_PATH),
      threshold40_audit_sha256: thresholdAuditHash,
      source_result_files_sha256: sourceResultHashes,
      source_manifest_files_sha256: sourceManifestHashes,
      included_identities_path: rel(includedPath),
      included_identities_sha256: includedHash,
      excluded_identities_path: rel(excludedPath),
      excluded_identities_sha256: excludedHash,
      projection_files_sha256: projectionHashes,
      supervisor_tables_path: rel(supervisorPath),
      supervisor_tables_sha256: supervisorHash,
      cohort_summary_csv_path: rel(summaryCsvPath),
      cohort_summary_csv_sha256: summaryCsvHash,
      protected_source_artifacts_sha256: beforeHashes,
    },
    validation: {
      plain_roll_base_identity_sets_equal: true,
      base_identity_count_matches_threshold40_audit: true,
      included_identity_count_matches_threshold40_audit: true,
      included_subject_count_matches_threshold40_audit: true,
      included_finger_distribution_matches_threshold40_audit: true,
      all_six_source_results_match_manifests_by_ordered_pair_id_and_identity: true,
      all_six_projection_identity_sets_equal: true,
      all_six_projection_row_counts_equal: true,
      all_projection_pair_ids_unique: true,
      projection_rows_identical_to_source_rows: true,
      projection_scores_timings_statuses_and_metadata_identical_to_source: true,
      all_four_self_projection_scores_at_least_40: true,
      all_four_self_projection_rejected_counts_zero: true,
      all_four_self_projection_accepted_percentages_100: true,
      excluded_only_because_of_plain_roll_score_count: 0,
      plain_roll_failures_removed_from_cohort: false,
      protected_source_artifacts_unchanged: true,
      dataset_files_loaded: false,
      dataset_files_written: false,
      sourceafis_rerun_performed: false,
      java_sidecar_started: false,
      benchmark_runner_started: false,
    },
  };
  const summaryJsonPath = path.join(OUTPUT_ROOT, "cohort_summary.json");
  await writeAtomic(summaryJsonPath, `${JSON.stringify(summary, null, 2)}\n`);

  const expectedOutputPaths = [
    includedPath,
    excludedPath,
    summaryCsvPath,
    summaryJsonPath,
    supervisorPath,
    ...CONDITIONS.map(([dataset, protocol]) => projectionPath(dataset, protocol)),
  ].map(rel).sort((left, right) => left.localeCompare(right, "en"));
  const actualOutputPaths = (await collectFiles(OUTPUT_ROOT)).map(rel).sort((left, right) => left.localeCompare(right, "en"));
  assert(arraysEqual(expectedOutputPaths, actualOutputPaths), "Output file set differs from the required deterministic cohort artifacts");
  const outputHashes = await hashFiles(await collectFiles(OUTPUT_ROOT));
  process.stdout.write(`${JSON.stringify({
    output_root: rel(OUTPUT_ROOT),
    base_identity_count: identities.length,
    included_identity_count: included.length,
    excluded_identity_count: excluded.length,
    included_subject_count: includedSubjects.size,
    included_canonical_finger_distribution: includedFingerDistribution,
    exclusion_count_by_reason_flag: exclusionCounts,
    excluded_identity_count_with_multiple_reasons: multipleReasonCount,
    exclusion_count_by_exact_reason_combination: orderedExactCombinationCounts,
    exclusion_pairwise_intersection_counts: pairwiseIntersections,
    plain_roll_summaries: plainRollSummaries,
    condition_summaries: summaries,
    output_sha256: outputHashes,
  }, null, 2)}\n`);
}

await main();
