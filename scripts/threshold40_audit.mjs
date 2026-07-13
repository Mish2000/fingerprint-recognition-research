import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const ROOT = path.resolve(path.dirname(SCRIPT_PATH), "..");
const THRESHOLD = 40.0;
const BUNDLE_ID = "cb6ed29d0231c44a3d95e60fd1b9fd7aa8f2fa333c8ccecd971b252c041830c3";
const COHORT_NAME = "sourceafis_joint_self_positive_v1";
const COHORT_ROOT = path.join(ROOT, "results", "cohorts", COHORT_NAME);
const ZERO_AUDIT_ROOT = path.join(ROOT, "results", "sourceafis", "pairwise-benchmark-v2", "zero_score_audit");
const OUTPUT_ROOT = path.join(ROOT, "results", "sourceafis", "pairwise-benchmark-v2", "threshold40_audit");

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
const RESULT_STATUSES = new Set([
  "ok", "prepare_a_failure", "prepare_b_failure", "comparison_failure",
]);

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

function percent(count, total) {
  return total === 0 ? 0 : rounded((count / total) * 100);
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
    assert(fields.length === headers.length, `Column count mismatch at row ${rowIndex + 2} in ${rel(filePath)}`);
    return { ...Object.fromEntries(headers.map((header, column) => [header, fields[column]])), rawRecord };
  });
  return { filePath, text, headers, rows };
}

async function sha256(filePath) {
  const data = await fs.readFile(filePath);
  return crypto.createHash("sha256").update(data).digest("hex");
}

function sha256Text(text) {
  return crypto.createHash("sha256").update(text, "utf8").digest("hex");
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function resultPath(dataset, protocol) {
  return path.join(ROOT, "results", dataset, protocol, "sourceafis", "pairwise-benchmark-v2", BUNDLE_ID, "pairs.csv");
}

function metadataPath(dataset, protocol) {
  return path.join(path.dirname(resultPath(dataset, protocol)), "run_metadata.json");
}

function manifestPath(dataset, protocol) {
  return path.join(ROOT, "protocols", dataset, `${protocol}.csv`);
}

function cohortPath(dataset, protocol) {
  return path.join(COHORT_ROOT, dataset, protocol, "pairs.csv");
}

async function collectFiles(directory) {
  const output = [];
  for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
    const candidate = path.join(directory, entry.name);
    if (entry.isDirectory()) output.push(...await collectFiles(candidate));
    else if (entry.isFile()) output.push(candidate);
  }
  return output.sort((left, right) => rel(left).localeCompare(rel(right), "en"));
}

async function hashFiles(filePaths) {
  return Object.fromEntries(await Promise.all(filePaths.map(async (filePath) => [rel(filePath), await sha256(filePath)])));
}

function requiredFinite(raw, label, { nonnegative = false } = {}) {
  assert(raw !== "", `Missing ${label}`);
  const value = Number(raw);
  assert(Number.isFinite(value), `Non-finite ${label}: ${raw}`);
  assert(!nonnegative || value >= 0, `Negative ${label}: ${raw}`);
  return value;
}

function validateResultRows(table, manifest, metadata, dataset, protocol) {
  const key = conditionKey(dataset, protocol);
  assert(arraysEqual(table.headers, RESULT_COLUMNS), `Result schema mismatch in ${key}`);
  assert(arraysEqual(manifest.headers, MANIFEST_COLUMNS), `Manifest schema mismatch in ${key}`);
  assert(table.rows.length === manifest.rows.length, `Result/manifest row count mismatch in ${key}`);
  const pairIds = new Set();
  const identities = new Set();
  let successCount = 0;
  const failureCounts = { prepare_a_failure: 0, prepare_b_failure: 0, comparison_failure: 0 };
  for (let index = 0; index < table.rows.length; index += 1) {
    const row = table.rows[index];
    const source = manifest.rows[index];
    for (const field of ["pair_id", "dataset", "protocol", "subject_id", "canonical_finger_position"]) {
      assert(row[field] === source[field], `Result/manifest ${field} mismatch at row ${index + 2} in ${key}`);
    }
    assert(row.dataset === dataset && row.protocol === protocol, `Wrong condition at row ${index + 2} in ${key}`);
    assert(!pairIds.has(row.pair_id), `Duplicate pair_id ${row.pair_id} in ${key}`);
    pairIds.add(row.pair_id);
    const identity = identityKey(row.subject_id, row.canonical_finger_position);
    assert(!identities.has(identity), `Duplicate identity ${identity} in ${key}`);
    identities.add(identity);
    for (const [field, expected] of Object.entries({
      method: metadata.method,
      method_version: metadata.method_version,
      benchmark_contract_version: metadata.benchmark_contract_version,
      result_schema_version: metadata.result_schema_version,
      config_hash: metadata.config_hash,
      implementation_hash: metadata.implementation_hash,
      manifest_sha256: metadata.manifest.sha256,
      score_direction: metadata.score_direction,
      score_semantics: metadata.score_semantics,
    })) assert(row[field] === expected, `${field} mismatch for ${row.pair_id}`);
    assert(RESULT_STATUSES.has(row.status), `Invalid status for ${row.pair_id}`);
    if (row.status === "ok") {
      requiredFinite(row.raw_score, `raw_score for ${row.pair_id}`);
      assert(row.error_code === "" && row.error_message === "", `Successful row contains error fields: ${row.pair_id}`);
      successCount += 1;
    } else {
      assert(row.raw_score === "", `Non-ok row contains raw_score: ${row.pair_id}`);
      assert(row.error_code.trim() && row.error_message.trim(), `Non-ok row lacks error fields: ${row.pair_id}`);
      failureCounts[row.status] += 1;
    }
    const executed = {
      prepare_a: true,
      prepare_b: row.status !== "prepare_a_failure",
      compare: row.status === "ok" || row.status === "comparison_failure",
    };
    const wall = [];
    const components = [];
    for (const [operation, wasExecuted] of Object.entries(executed)) {
      const wallField = `${operation}_ms`;
      const methodField = `method_${operation}_ms`;
      const diagnosticsField = `${operation}_diagnostics`;
      if (!wasExecuted) {
        assert(row[wallField] === "" && row[methodField] === "" && row[diagnosticsField] === "", `Unexpected ${operation} output for ${row.pair_id}`);
        continue;
      }
      const wallValue = requiredFinite(row[wallField], `${wallField} for ${row.pair_id}`, { nonnegative: true });
      wall.push(wallValue);
      components.push(wallValue);
      if (row[methodField] !== "") components.push(requiredFinite(row[methodField], `${methodField} for ${row.pair_id}`, { nonnegative: true }));
      assert(row[diagnosticsField] !== "", `Missing ${diagnosticsField} for ${row.pair_id}`);
      JSON.parse(row[diagnosticsField]);
    }
    const total = requiredFinite(row.total_ms, `total_ms for ${row.pair_id}`, { nonnegative: true });
    assert(components.every((component) => total + 0.001 >= component), `total_ms smaller than component for ${row.pair_id}`);
    if (row.status === "ok") assert(total + 0.001 >= wall.reduce((sum, value) => sum + value, 0), `total_ms smaller than wall sum for ${row.pair_id}`);
  }
  assert(metadata.success_count === successCount, `Metadata success_count mismatch in ${key}`);
  for (const [status, count] of Object.entries(failureCounts)) assert(metadata.failure_counts[status] === count, `Metadata ${status} count mismatch in ${key}`);
  return { pairIds, identities };
}

async function validatePrimary(dataset, protocol) {
  const key = conditionKey(dataset, protocol);
  const pairsPath = resultPath(dataset, protocol);
  const metaPath = metadataPath(dataset, protocol);
  const maniPath = manifestPath(dataset, protocol);
  const [table, manifest, metadataText] = await Promise.all([
    readCsv(pairsPath), readCsv(maniPath), fs.readFile(metaPath, "utf8"),
  ]);
  const metadata = JSON.parse(metadataText);
  assert(metadata.dataset === dataset && metadata.protocol === protocol, `Metadata condition mismatch in ${key}`);
  assert(metadata.method === "sourceafis", `Unexpected method in ${key}`);
  assert(metadata.benchmark_contract_version === "pairwise-benchmark-v2", `Wrong benchmark contract in ${key}`);
  assert(metadata.result_schema_version === "pairwise-result-v2", `Wrong result schema in ${key}`);
  assert(metadata.config_hash === BUNDLE_ID, `Wrong config hash in ${key}`);
  assert(await sha256(pairsPath) === metadata.result.sha256, `Result SHA mismatch in ${key}`);
  assert(await sha256(maniPath) === metadata.manifest.sha256, `Manifest SHA mismatch in ${key}`);
  assert(metadata.result.row_count === table.rows.length, `Metadata result row count mismatch in ${key}`);
  assert(metadata.manifest.row_count === manifest.rows.length, `Metadata manifest row count mismatch in ${key}`);
  assert(path.resolve(metadata.manifest.path) === path.resolve(maniPath), `Metadata manifest path mismatch in ${key}`);
  const validated = validateResultRows(table, manifest, metadata, dataset, protocol);
  const payload = table.rows.map((row) => ({
    error_code: row.error_code,
    pair_id: row.pair_id,
    raw_score: row.raw_score,
    status: row.status,
  }));
  assert(sha256Text(canonicalJson(payload)) === metadata.result.score_payload_sha256, `Score payload SHA mismatch in ${key}`);
  table.rows = table.rows.map((row) => ({
    ...row,
    key: identityKey(row.subject_id, row.canonical_finger_position),
    score: row.status === "ok" ? Number(row.raw_score) : null,
  }));
  table.rowMap = new Map(table.rows.map((row) => [row.key, row]));
  table.pairMap = new Map(table.rows.map((row) => [row.pair_id, row]));
  return { table, manifest, metadata, validated };
}

async function validateCohort(primary, inputHashes) {
  const summaryPath = path.join(COHORT_ROOT, "cohort_summary.json");
  const summary = JSON.parse(await fs.readFile(summaryPath, "utf8"));
  assert(summary.cohort_name === COHORT_NAME, "Wrong cohort name");
  assert(summary.source_bundle_config_hash === BUNDLE_ID, "Wrong cohort source bundle");
  const provenance = summary.provenance;
  for (const [filePath, expected] of Object.entries({
    ...provenance.source_result_files_sha256,
    ...provenance.source_manifest_files_sha256,
    ...provenance.zero_score_audit_artifacts_sha256,
    [rel(path.join(COHORT_ROOT, "included_identities.csv"))]: provenance.included_identities_sha256,
    ...provenance.filtered_output_files_sha256,
    ...provenance.other_deterministic_output_files_sha256,
  })) assert(inputHashes[filePath] === expected, `Cohort provenance SHA mismatch: ${filePath}`);

  const cohort = new Map();
  let commonIdentityKeys = null;
  for (const [dataset, protocol] of CONDITIONS) {
    const key = conditionKey(dataset, protocol);
    const table = await readCsv(cohortPath(dataset, protocol));
    assert(arraysEqual(table.headers, RESULT_COLUMNS), `Cohort result schema mismatch in ${key}`);
    const pairIds = new Set();
    const identities = new Set();
    const source = primary.get(key).table;
    for (const row of table.rows) {
      assert(row.dataset === dataset && row.protocol === protocol, `Wrong cohort condition in ${key}`);
      assert(!pairIds.has(row.pair_id), `Duplicate cohort pair_id in ${key}: ${row.pair_id}`);
      pairIds.add(row.pair_id);
      const identity = identityKey(row.subject_id, row.canonical_finger_position);
      assert(!identities.has(identity), `Duplicate cohort identity in ${key}: ${identity}`);
      identities.add(identity);
      assert(row.status === "ok", `Existing cohort contains non-ok row in ${key}: ${row.pair_id}`);
      const score = requiredFinite(row.raw_score, `cohort raw_score for ${row.pair_id}`);
      const sourceRow = source.pairMap.get(row.pair_id);
      assert(sourceRow && sourceRow.rawRecord === row.rawRecord, `Cohort row differs from source in ${key}: ${row.pair_id}`);
      row.key = identity;
      row.score = score;
    }
    const sortedKeys = [...identities].sort();
    if (commonIdentityKeys === null) commonIdentityKeys = sortedKeys;
    else assert(arraysEqual(commonIdentityKeys, sortedKeys), `Cohort identity set mismatch in ${key}`);
    table.rowMap = new Map(table.rows.map((row) => [row.key, row]));
    cohort.set(key, table);
    const conditionSummary = summary.condition_summaries.find((item) => item.dataset === dataset && item.protocol === protocol);
    assert(conditionSummary.filtered_row_count === table.rows.length, `Cohort summary row count mismatch in ${key}`);
    assert(conditionSummary.filtered_output_sha256 === inputHashes[rel(table.filePath)], `Cohort summary output SHA mismatch in ${key}`);
  }

  const included = await readCsv(path.join(COHORT_ROOT, "included_identities.csv"));
  const expectedIncludedHeaders = [
    "subject_id", "canonical_finger_position",
    ...CONDITIONS.map(([dataset, protocol]) => `${dataset}_${protocol}_raw_score`),
  ];
  assert(arraysEqual(included.headers, expectedIncludedHeaders), "Included identities schema mismatch");
  assert(included.rows.length === commonIdentityKeys.length, "Included identities row count mismatch");
  const includedKeys = new Set();
  for (const row of included.rows) {
    const key = identityKey(row.subject_id, row.canonical_finger_position);
    assert(!includedKeys.has(key), `Duplicate included identity: ${key}`);
    includedKeys.add(key);
    for (const [dataset, protocol] of CONDITIONS) {
      const condition = conditionKey(dataset, protocol);
      const scoreColumn = `${dataset}_${protocol}_raw_score`;
      assert(cohort.get(condition).rowMap.get(key).raw_score === row[scoreColumn], `Included score mismatch for ${key} in ${condition}`);
    }
  }
  assert(arraysEqual([...includedKeys].sort(), commonIdentityKeys), "Included identities differ from cohort pairs");

  const excluded = await readCsv(path.join(COHORT_ROOT, "excluded_identities.csv"));
  const excludedKeys = new Set();
  for (const row of excluded.rows) {
    const key = identityKey(row.subject_id, row.canonical_finger_position);
    assert(!excludedKeys.has(key) && !includedKeys.has(key), `Invalid excluded identity: ${key}`);
    excludedKeys.add(key);
    assert(row.reason_flags.trim() !== "", `Excluded identity lacks reason: ${key}`);
    for (const [dataset, protocol] of CONDITIONS) {
      const sourceRow = primary.get(conditionKey(dataset, protocol)).table.rowMap.get(key);
      const scoreColumn = `${dataset}_${protocol}_raw_score`;
      assert((sourceRow?.raw_score ?? "") === row[scoreColumn], `Excluded score mismatch for ${key} in ${dataset}/${protocol}`);
    }
  }
  const baseKeys = [...primary.get("sd300b/plain_roll").table.rowMap.keys()].sort();
  assert(arraysEqual(baseKeys, [...primary.get("sd300c/plain_roll").table.rowMap.keys()].sort()), "Primary plain_roll identity sets differ");
  assert(arraysEqual(baseKeys, [...includedKeys, ...excludedKeys].sort()), "Included/excluded cohort partition does not equal base population");
  assert(summary.counts.included_identity_count === includedKeys.size, "Cohort included count mismatch");
  assert(summary.counts.excluded_identity_count === excludedKeys.size, "Cohort excluded count mismatch");

  const summaryCsv = await readCsv(path.join(COHORT_ROOT, "cohort_summary.csv"));
  assert(summaryCsv.rows.length === CONDITIONS.length, "Cohort summary CSV row count mismatch");
  return { cohort, summary, included, excluded, commonIdentityKeys };
}

function bucketSummary(scope, dataset, protocol, table) {
  const ok = table.rows.filter((row) => row.status === "ok");
  const nonOk = table.rows.length - ok.length;
  const zero = ok.filter((row) => row.score === 0).length;
  const positiveBelow = ok.filter((row) => row.score > 0 && row.score < THRESHOLD).length;
  const accepted = ok.filter((row) => row.score >= THRESHOLD).length;
  const rejected = ok.filter((row) => row.score < THRESHOLD).length;
  assert(accepted + rejected + nonOk === table.rows.length, `Decision reconciliation failed in ${scope} ${dataset}/${protocol}`);
  assert(zero <= rejected, `Zero score is not a rejected subset in ${scope} ${dataset}/${protocol}`);
  assert(zero + positiveBelow === rejected, `Rejected buckets do not reconcile in ${scope} ${dataset}/${protocol}`);
  return {
    scope, dataset, protocol, threshold: THRESHOLD, total_count: table.rows.length,
    ok_count: ok.length, non_ok_count: nonOk, score_zero_count: zero,
    score_positive_below_threshold_count: positiveBelow, accepted_count: accepted,
    rejected_count: rejected, accepted_percentage: percent(accepted, table.rows.length),
    rejected_percentage: percent(rejected, table.rows.length),
  };
}

function plainRollDecision(summary) {
  return {
    scope: summary.scope,
    dataset: summary.dataset,
    protocol: summary.protocol,
    threshold: summary.threshold,
    total_genuine_pairs: summary.total_count,
    genuine_accept_count: summary.accepted_count,
    false_non_match_count: summary.rejected_count,
    false_non_match_score_zero_count: summary.score_zero_count,
    false_non_match_positive_below_threshold_count: summary.score_positive_below_threshold_count,
    non_ok_count: summary.non_ok_count,
    genuine_accept_percentage: summary.accepted_percentage,
    false_non_match_percentage: summary.rejected_percentage,
  };
}

function pairedAnalysis(scope, bTable, cTable) {
  const bKeys = [...bTable.rowMap.keys()].sort();
  const cKeys = [...cTable.rowMap.keys()].sort();
  assert(arraysEqual(bKeys, cKeys), `Paired B/C identity sets differ in ${scope}`);
  const counts = {
    total_paired_identity_count: bKeys.length,
    ok_in_both_count: 0,
    non_ok_in_either_count: 0,
    accepted_in_both_count: 0,
    rejected_in_both_count: 0,
    accepted_only_in_sd300b_count: 0,
    accepted_only_in_sd300c_count: 0,
    score_zero_in_both_count: 0,
    score_zero_only_in_one_count: 0,
    positive_below_threshold_in_both_count: 0,
    decision_disagreement_count: 0,
    decision_agreement_count: 0,
  };
  const rows = [];
  for (const key of bKeys) {
    const b = bTable.rowMap.get(key);
    const c = cTable.rowMap.get(key);
    const bothOk = b.status === "ok" && c.status === "ok";
    if (bothOk) counts.ok_in_both_count += 1;
    else counts.non_ok_in_either_count += 1;
    const bAccepted = bothOk && b.score >= THRESHOLD;
    const cAccepted = bothOk && c.score >= THRESHOLD;
    const acceptedBoth = bAccepted && cAccepted;
    const rejectedBoth = bothOk && !bAccepted && !cAccepted;
    const acceptedOnlyB = bAccepted && !cAccepted;
    const acceptedOnlyC = cAccepted && !bAccepted;
    const zeroBoth = bothOk && b.score === 0 && c.score === 0;
    const zeroOnlyOne = bothOk && ((b.score === 0) !== (c.score === 0));
    const positiveBelowBoth = bothOk && b.score > 0 && b.score < THRESHOLD && c.score > 0 && c.score < THRESHOLD;
    const disagreement = acceptedOnlyB || acceptedOnlyC;
    const agreement = acceptedBoth || rejectedBoth;
    if (acceptedBoth) counts.accepted_in_both_count += 1;
    if (rejectedBoth) counts.rejected_in_both_count += 1;
    if (acceptedOnlyB) counts.accepted_only_in_sd300b_count += 1;
    if (acceptedOnlyC) counts.accepted_only_in_sd300c_count += 1;
    if (zeroBoth) counts.score_zero_in_both_count += 1;
    if (zeroOnlyOne) counts.score_zero_only_in_one_count += 1;
    if (positiveBelowBoth) counts.positive_below_threshold_in_both_count += 1;
    if (disagreement) counts.decision_disagreement_count += 1;
    if (agreement) counts.decision_agreement_count += 1;
    rows.push({
      scope,
      subject_id: b.subject_id,
      canonical_finger_position: Number(b.canonical_finger_position),
      sd300b_status: b.status,
      sd300b_raw_score: b.raw_score,
      sd300b_decision: b.status === "ok" ? (bAccepted ? "genuine_accept" : "false_non_match") : "non_ok",
      sd300c_status: c.status,
      sd300c_raw_score: c.raw_score,
      sd300c_decision: c.status === "ok" ? (cAccepted ? "genuine_accept" : "false_non_match") : "non_ok",
      accepted_in_both: acceptedBoth,
      rejected_in_both: rejectedBoth,
      accepted_only_in_sd300b: acceptedOnlyB,
      accepted_only_in_sd300c: acceptedOnlyC,
      score_zero_in_both: zeroBoth,
      score_zero_only_in_one: zeroOnlyOne,
      positive_below_threshold_in_both: positiveBelowBoth,
      decision_disagreement: disagreement,
    });
  }
  assert(counts.accepted_in_both_count + counts.rejected_in_both_count + counts.decision_disagreement_count + counts.non_ok_in_either_count === counts.total_paired_identity_count, `Paired decision reconciliation failed in ${scope}`);
  counts.decision_agreement_percentage = percent(counts.decision_agreement_count, counts.total_paired_identity_count);
  counts.decision_disagreement_percentage = percent(counts.decision_disagreement_count, counts.total_paired_identity_count);
  return { scope, threshold: THRESHOLD, ...counts, rows };
}

async function writeAtomic(filePath, content) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const tempPath = `${filePath}.tmp`;
  await fs.writeFile(tempPath, content, "utf8");
  await fs.rename(tempPath, filePath);
}

async function main() {
  assert(path.resolve(OUTPUT_ROOT).startsWith(path.resolve(ROOT) + path.sep), "Unsafe output path");
  const primaryFiles = CONDITIONS.flatMap(([dataset, protocol]) => [
    resultPath(dataset, protocol), metadataPath(dataset, protocol), manifestPath(dataset, protocol),
  ]);
  const cohortFiles = await collectFiles(COHORT_ROOT);
  const zeroAuditFiles = await collectFiles(ZERO_AUDIT_ROOT);
  const protectedFiles = [...primaryFiles, ...cohortFiles, ...zeroAuditFiles]
    .sort((left, right) => rel(left).localeCompare(rel(right), "en"));
  const beforeHashes = await hashFiles(protectedFiles);

  const primary = new Map();
  for (const [dataset, protocol] of CONDITIONS) {
    primary.set(conditionKey(dataset, protocol), await validatePrimary(dataset, protocol));
  }
  const cohortValidation = await validateCohort(primary, beforeHashes);

  const summaries = [];
  for (const scope of ["primary_full_results", "self_positive_cohort"]) {
    for (const [dataset, protocol] of CONDITIONS) {
      const table = scope === "primary_full_results"
        ? primary.get(conditionKey(dataset, protocol)).table
        : cohortValidation.cohort.get(conditionKey(dataset, protocol));
      summaries.push(bucketSummary(scope, dataset, protocol, table));
    }
  }

  const selfSets = Object.fromEntries(SELF_KEYS.map((key) => [
    key,
    new Set(cohortValidation.cohort.get(key).rows.filter((row) => row.score > 0 && row.score < THRESHOLD).map((row) => row.key)),
  ]));
  const unionKeys = [...new Set(SELF_KEYS.flatMap((key) => [...selfSets[key]]))].sort();
  const intersectionKeys = unionKeys.filter((key) => SELF_KEYS.every((condition) => selfSets[condition].has(key)));
  const cohortBaseKeys = cohortValidation.commonIdentityKeys;
  const acceptedAllKeys = cohortBaseKeys.filter((key) => SELF_KEYS.every((condition) => {
    const row = cohortValidation.cohort.get(condition).rowMap.get(key);
    return row.status === "ok" && row.score >= THRESHOLD;
  }));
  const acceptedAllRows = acceptedAllKeys.map((key) => cohortValidation.cohort.get(SELF_KEYS[0]).rowMap.get(key));
  const distinctSubjects = new Set(acceptedAllRows.map((row) => row.subject_id));
  const fingerDistribution = Object.fromEntries(Array.from({ length: 10 }, (_, index) => index + 1).map((position) => [
    String(position), acceptedAllRows.filter((row) => Number(row.canonical_finger_position) === position).length,
  ]));
  const selfBelowRows = unionKeys.map((key) => {
    const reference = cohortValidation.cohort.get(SELF_KEYS[0]).rowMap.get(key);
    const scores = Object.fromEntries(SELF_KEYS.map((condition) => [condition, cohortValidation.cohort.get(condition).rowMap.get(key).raw_score]));
    const flags = Object.fromEntries(SELF_KEYS.map((condition) => [condition, selfSets[condition].has(key)]));
    return { reference, scores, flags, failingCount: SELF_KEYS.filter((condition) => flags[condition]).length };
  }).sort((left, right) => compareIdentity(left.reference, right.reference));

  const plainRollDecisions = summaries
    .filter((summary) => summary.protocol === "plain_roll")
    .map(plainRollDecision);
  const paired = [
    pairedAnalysis("primary_full_results", primary.get("sd300b/plain_roll").table, primary.get("sd300c/plain_roll").table),
    pairedAnalysis("self_positive_cohort", cohortValidation.cohort.get("sd300b/plain_roll"), cohortValidation.cohort.get("sd300c/plain_roll")),
  ];

  const summaryHeaders = [
    "scope", "dataset", "protocol", "threshold", "total_count", "ok_count", "non_ok_count",
    "score_zero_count", "score_positive_below_threshold_count", "accepted_count", "rejected_count",
    "accepted_percentage", "rejected_percentage",
  ];
  const summaryPath = path.join(OUTPUT_ROOT, "threshold40_summary.csv");
  await writeAtomic(summaryPath, csvText(summaryHeaders, summaries.map((row) => summaryHeaders.map((header) => row[header]))));

  const selfHeaders = [
    "subject_id", "canonical_finger_position",
    "sd300b_plain_self_raw_score", "sd300b_plain_self_positive_below_threshold",
    "sd300b_roll_self_raw_score", "sd300b_roll_self_positive_below_threshold",
    "sd300c_plain_self_raw_score", "sd300c_plain_self_positive_below_threshold",
    "sd300c_roll_self_raw_score", "sd300c_roll_self_positive_below_threshold",
    "below_threshold_self_condition_count",
  ];
  const selfPath = path.join(OUTPUT_ROOT, "self_below_threshold_identities.csv");
  await writeAtomic(selfPath, csvText(selfHeaders, selfBelowRows.map(({ reference, scores, flags, failingCount }) => [
    reference.subject_id, reference.canonical_finger_position,
    scores["sd300b/plain_self"], flags["sd300b/plain_self"],
    scores["sd300b/roll_self"], flags["sd300b/roll_self"],
    scores["sd300c/plain_self"], flags["sd300c/plain_self"],
    scores["sd300c/roll_self"], flags["sd300c/roll_self"],
    failingCount,
  ])));

  const plainHeaders = [
    "scope", "dataset", "protocol", "threshold", "total_genuine_pairs", "genuine_accept_count",
    "false_non_match_count", "false_non_match_score_zero_count",
    "false_non_match_positive_below_threshold_count", "non_ok_count",
    "genuine_accept_percentage", "false_non_match_percentage",
  ];
  const plainPath = path.join(OUTPUT_ROOT, "plain_roll_decisions.csv");
  await writeAtomic(plainPath, csvText(plainHeaders, plainRollDecisions.map((row) => plainHeaders.map((header) => row[header]))));

  const pairedHeaders = [
    "scope", "subject_id", "canonical_finger_position", "sd300b_status", "sd300b_raw_score",
    "sd300b_decision", "sd300c_status", "sd300c_raw_score", "sd300c_decision",
    "accepted_in_both", "rejected_in_both", "accepted_only_in_sd300b", "accepted_only_in_sd300c",
    "score_zero_in_both", "score_zero_only_in_one", "positive_below_threshold_in_both",
    "decision_disagreement",
  ];
  const pairedPath = path.join(OUTPUT_ROOT, "paired_bc_decisions.csv");
  await writeAtomic(pairedPath, csvText(pairedHeaders, paired.flatMap((analysis) => analysis.rows.map((row) => pairedHeaders.map((header) => row[header])))));

  const pairedSummaryHeaders = [
    "scope", "threshold", "total_paired_identity_count", "ok_in_both_count", "non_ok_in_either_count",
    "accepted_in_both_count", "rejected_in_both_count", "accepted_only_in_sd300b_count",
    "accepted_only_in_sd300c_count", "score_zero_in_both_count", "score_zero_only_in_one_count",
    "positive_below_threshold_in_both_count", "decision_agreement_count", "decision_disagreement_count",
    "decision_agreement_percentage", "decision_disagreement_percentage",
  ];
  const pairedSummaryPath = path.join(OUTPUT_ROOT, "paired_bc_decision_summary.csv");
  await writeAtomic(pairedSummaryPath, csvText(pairedSummaryHeaders, paired.map((analysis) => pairedSummaryHeaders.map((header) => analysis[header]))));

  const csvOutputs = [summaryPath, selfPath, plainPath, pairedPath, pairedSummaryPath];
  const csvHashes = await hashFiles(csvOutputs);
  const afterHashes = await hashFiles(protectedFiles);
  assert(canonicalJson(beforeHashes) === canonicalJson(afterHashes), "A protected source artifact changed during the audit");

  const audit = {
    audit_schema_version: "sourceafis-threshold-decision-audit-v1",
    sourceafis_threshold: THRESHOLD,
    descriptive_audit_only: true,
    definitions: {
      self_positive: "raw_score > 0",
      self_accepted_at_threshold_40: "raw_score >= 40",
      existing_cohort_label_constraint: "The existing cohort must not be described as 100% matched unless every identity is accepted in all four self protocols at threshold 40.",
      score_zero: "status == ok and raw_score == 0",
      score_positive_below_threshold: "status == ok and 0 < raw_score < 40",
      accepted: "status == ok and raw_score >= 40",
      rejected: "status == ok and raw_score < 40",
      percentage_denominator: "total_count, with non-ok rows reported separately and excluded from score buckets",
      plain_roll_semantics: "All plain_roll pairs are genuine; accepted rows are genuine accepts and rejected rows are false non-matches. False match and true reject terminology is not used.",
    },
    input_artifacts_sha256: beforeHashes,
    primary_bundle_validation: {
      all_six_source_bundles_validated_before_calculation_by_audit_validator: true,
      all_six_result_schemas_and_metadata_valid: true,
      all_six_results_match_manifests_by_ordered_pair_id_and_identity: true,
      all_six_result_and_manifest_hashes_match_metadata: true,
      all_six_score_payload_hashes_match_metadata: true,
    },
    existing_cohort_validation: {
      cohort_summary_provenance_hashes_match: true,
      all_six_filtered_files_are_exact_source_row_subsets: true,
      all_six_filtered_identity_sets_equal: true,
      included_identities_scores_match_all_six_filtered_files: true,
      included_and_excluded_identities_partition_plain_roll_base: true,
      cohort_row_count: cohortValidation.commonIdentityKeys.length,
    },
    threshold_summaries: summaries,
    self_protocol_audit: {
      scope: "self_positive_cohort",
      positive_below_threshold_identity_count_by_condition: Object.fromEntries(SELF_KEYS.map((key) => [key, selfSets[key].size])),
      positive_below_threshold_identity_union_count: unionKeys.length,
      positive_below_threshold_identity_intersection_count: intersectionKeys.length,
      accepted_in_all_four_self_protocols_count: acceptedAllKeys.length,
      not_accepted_in_all_four_self_protocols_count: cohortBaseKeys.length - acceptedAllKeys.length,
      possible_threshold40_cohort_identity_count: acceptedAllKeys.length,
      possible_threshold40_cohort_distinct_subject_count: distinctSubjects.size,
      possible_threshold40_cohort_canonical_finger_distribution: fingerDistribution,
      existing_cohort_has_100_percent_self_acceptance_at_threshold_40: acceptedAllKeys.length === cohortBaseKeys.length,
      new_cohort_created: false,
    },
    plain_roll_decisions: plainRollDecisions,
    paired_bc_decision_summaries: paired.map(({ rows, ...summary }) => summary),
    deterministic_csv_outputs_sha256: csvHashes,
    validation: {
      every_source_row_count_matches_manifest_and_metadata: true,
      every_nonblank_score_is_finite: true,
      every_summary_reconciles_accepted_plus_rejected_plus_non_ok_to_total: true,
      every_zero_score_count_is_a_subset_of_rejected_count: true,
      protected_source_artifacts_unchanged_during_audit: true,
      dataset_files_written_by_audit: false,
      output_contains_timestamps: false,
      sourceafis_rerun_performed: false,
      java_sidecar_started: false,
      benchmark_runner_started: false,
      calibration_performed: false,
      probabilities_computed: false,
      impostor_pairs_created: false,
      threshold40_cohort_created: false,
      plain_roll_used_for_cohort_filtering: false,
    },
  };
  const auditPath = path.join(OUTPUT_ROOT, "threshold40_audit.json");
  await writeAtomic(auditPath, `${JSON.stringify(audit, null, 2)}\n`);
  const outputHashes = await hashFiles([...csvOutputs, auditPath]);
  process.stdout.write(`${JSON.stringify({
    output_root: rel(OUTPUT_ROOT),
    self_protocol_audit: audit.self_protocol_audit,
    plain_roll_decisions: plainRollDecisions,
    paired_bc_decision_summaries: audit.paired_bc_decision_summaries,
    output_sha256: outputHashes,
  }, null, 2)}\n`);
}

await main();
