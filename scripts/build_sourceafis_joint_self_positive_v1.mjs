import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const ROOT = path.resolve(path.dirname(SCRIPT_PATH), "..");
const COHORT_NAME = "sourceafis_joint_self_positive_v1";
const COHORT_VERSION = "v1";
const COHORT_DESCRIPTION = "SourceAFIS-selected joint self-positive cohort";
const BUNDLE_ID = "cb6ed29d0231c44a3d95e60fd1b9fd7aa8f2fa333c8ccecd971b252c041830c3";
const OUTPUT_ROOT = path.join(ROOT, "results", "cohorts", COHORT_NAME);
const REASONS = [
  "sd300b_plain_self_zero",
  "sd300b_roll_self_zero",
  "sd300c_plain_self_zero",
  "sd300c_roll_self_zero",
  "missing_required_protocol_identity",
];
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
const COHORT_RULE = [
  "The identity unit is (subject_id, canonical_finger_position).",
  "The base population is the exact shared anatomical-identity set in the sd300b/plain_roll and sd300c/plain_roll SourceAFIS pairwise-benchmark-v2 result bundles; those two sets must be identical or construction stops.",
  "An identity is included only when it exists in sd300b/plain_self, sd300b/roll_self, sd300b/plain_roll, sd300c/plain_self, sd300c/roll_self, and sd300c/plain_roll, and its raw_score is greater than zero in each of sd300b/plain_self, sd300b/roll_self, sd300c/plain_self, and sd300c/roll_self.",
  "The plain_roll raw_score is never used for inclusion: an identity with plain_roll raw_score equal to zero remains included when all four self scores are positive.",
  "An identity with raw_score equal to zero in any of the four self conditions is excluded and receives every applicable zero-score reason flag; a required identity absent from any of the six conditions receives missing_required_protocol_identity.",
].join(" ");

function rel(filePath) {
  return path.relative(ROOT, filePath).split(path.sep).join("/");
}

async function sha256(filePath) {
  const data = await fs.readFile(filePath);
  return crypto.createHash("sha256").update(data).digest("hex");
}

function scanCsvRecords(text) {
  const records = [];
  let start = 0;
  let inQuotes = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (char === '"') {
      if (inQuotes && text[index + 1] === '"') {
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (!inQuotes && (char === "\n" || char === "\r")) {
      records.push(text.slice(start, index));
      if (char === "\r" && text[index + 1] === "\n") index += 1;
      start = index + 1;
    }
  }
  if (inQuotes) throw new Error("Unterminated quoted CSV field");
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
        } else {
          inQuotes = false;
        }
      } else {
        value += char;
      }
    } else if (char === ",") {
      fields.push(value);
      value = "";
    } else if (char === '"' && value === "") {
      inQuotes = true;
    } else {
      value += char;
    }
  }
  if (inQuotes) throw new Error("Unterminated quoted CSV field in record");
  fields.push(value);
  return fields;
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function csvText(headers, rows) {
  return `${[headers, ...rows].map((row) => row.map(csvEscape).join(",")).join("\n")}\n`;
}

function identityKey(subjectId, fingerPosition) {
  return `${subjectId}\u001f${Number(fingerPosition)}`;
}

function compareIdentityRows(left, right) {
  const subjectComparison = left.subject_id.localeCompare(right.subject_id, "en");
  return subjectComparison || Number(left.canonical_finger_position) - Number(right.canonical_finger_position);
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function arraysEqual(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function rounded(value) {
  return Number(value.toFixed(12));
}

function numberText(value) {
  return Number(value.toFixed(12)).toString();
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

async function readTable(filePath, expectedDataset, expectedProtocol) {
  const text = await fs.readFile(filePath, "utf8");
  const records = scanCsvRecords(text);
  assert(records.length >= 1, `Empty CSV: ${rel(filePath)}`);
  const headers = parseCsvRecord(records[0]);
  const index = Object.fromEntries(headers.map((header, column) => [header, column]));
  for (const required of ["pair_id", "dataset", "protocol", "subject_id", "canonical_finger_position"]) {
    assert(required in index, `Missing ${required} in ${rel(filePath)}`);
  }
  const rows = [];
  const pairIds = new Set();
  const identities = new Set();
  for (const rawRecord of records.slice(1)) {
    const fields = parseCsvRecord(rawRecord);
    assert(fields.length === headers.length, `Column count mismatch in ${rel(filePath)}`);
    const row = Object.fromEntries(headers.map((header, column) => [header, fields[column]]));
    assert(row.dataset === expectedDataset, `Wrong dataset in ${rel(filePath)}: ${row.dataset}`);
    assert(row.protocol === expectedProtocol, `Wrong protocol in ${rel(filePath)}: ${row.protocol}`);
    assert(!pairIds.has(row.pair_id), `Duplicate pair_id ${row.pair_id} in ${rel(filePath)}`);
    pairIds.add(row.pair_id);
    const key = identityKey(row.subject_id, row.canonical_finger_position);
    assert(!identities.has(key), `Duplicate anatomical identity ${key} in ${rel(filePath)}`);
    identities.add(key);
    rows.push({ ...row, key, rawRecord });
  }
  const eol = text.includes("\r\n") ? "\r\n" : "\n";
  return { filePath, text, records, headers, index, rows, pairIds, identities, eol, headerRecord: records[0] };
}

async function writeUtf8(filePath, content) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, content, "utf8");
}

async function main() {
  const auditRoot = path.join(ROOT, "results", "sourceafis", "pairwise-benchmark-v2", "zero_score_audit");
  const auditFiles = (await fs.readdir(auditRoot, { withFileTypes: true }))
    .filter((entry) => entry.isFile())
    .map((entry) => path.join(auditRoot, entry.name))
    .sort((left, right) => rel(left).localeCompare(rel(right), "en"));
  assert(auditFiles.length > 0, "No zero-score audit artifacts found");

  const sourceFiles = CONDITIONS.map(([dataset, protocol]) => resultPath(dataset, protocol));
  const manifestFiles = CONDITIONS.map(([dataset, protocol]) => manifestPath(dataset, protocol));
  const protectedFiles = [...sourceFiles, ...manifestFiles, ...auditFiles];
  const beforeHashes = Object.fromEntries(await Promise.all(protectedFiles.map(async (filePath) => [rel(filePath), await sha256(filePath)])));

  const tables = new Map();
  const manifests = new Map();
  for (const [dataset, protocol] of CONDITIONS) {
    const key = `${dataset}/${protocol}`;
    const table = await readTable(resultPath(dataset, protocol), dataset, protocol);
    for (const required of ["raw_score", "method_compare_ms", "manifest_sha256", "status"]) {
      assert(required in table.index, `Missing ${required} in ${rel(table.filePath)}`);
    }
    const manifest = await readTable(manifestPath(dataset, protocol), dataset, protocol);
    const actualManifestHash = beforeHashes[rel(manifest.filePath)];
    assert(table.rows.every((row) => row.manifest_sha256 === actualManifestHash), `Manifest hash mismatch in ${key}`);
    assert(arraysEqual([...table.pairIds].sort(), [...manifest.pairIds].sort()), `Result/manifest pair_id mismatch in ${key}`);
    assert(arraysEqual([...table.identities].sort(), [...manifest.identities].sort()), `Result/manifest identity mismatch in ${key}`);
    tables.set(key, table);
    manifests.set(key, manifest);
  }

  const baseB = [...tables.get("sd300b/plain_roll").identities].sort();
  const baseC = [...tables.get("sd300c/plain_roll").identities].sort();
  assert(arraysEqual(baseB, baseC), "Base population inconsistency: sd300b/plain_roll and sd300c/plain_roll identity sets differ");
  const baseKeys = baseB;

  const rowMaps = new Map([...tables].map(([key, table]) => [key, new Map(table.rows.map((row) => [row.key, row]))]));
  const identities = baseKeys.map((key) => {
    const first = rowMaps.get("sd300b/plain_roll").get(key);
    const scores = Object.fromEntries(CONDITIONS.map(([dataset, protocol]) => {
      const condition = `${dataset}/${protocol}`;
      const row = rowMaps.get(condition).get(key);
      return [condition, row ? Number(row.raw_score) : null];
    }));
    const missingRequired = CONDITIONS.some(([dataset, protocol]) => !rowMaps.get(`${dataset}/${protocol}`).has(key));
    const reasons = [];
    if (scores["sd300b/plain_self"] === 0) reasons.push("sd300b_plain_self_zero");
    if (scores["sd300b/roll_self"] === 0) reasons.push("sd300b_roll_self_zero");
    if (scores["sd300c/plain_self"] === 0) reasons.push("sd300c_plain_self_zero");
    if (scores["sd300c/roll_self"] === 0) reasons.push("sd300c_roll_self_zero");
    if (missingRequired) reasons.push("missing_required_protocol_identity");
    for (const selfKey of SELF_KEYS) {
      const score = scores[selfKey];
      if (score !== null) assert(Number.isFinite(score) && score >= 0, `Invalid self score for ${key} in ${selfKey}`);
    }
    const included = !missingRequired && SELF_KEYS.every((selfKey) => scores[selfKey] > 0);
    assert(included === (reasons.length === 0), `Exclusion reason coverage failure for ${key}`);
    return {
      key,
      subject_id: first.subject_id,
      canonical_finger_position: Number(first.canonical_finger_position),
      scores,
      reasons,
      included,
    };
  }).sort(compareIdentityRows);

  const included = identities.filter((identity) => identity.included);
  const excluded = identities.filter((identity) => !identity.included);
  const includedKeys = new Set(included.map((identity) => identity.key));
  assert(included.every((identity) => SELF_KEYS.every((key) => identity.scores[key] > 0)), "Included identity has a non-positive self score");
  assert(excluded.every((identity) => identity.reasons.length > 0), "Excluded identity lacks an allowed exclusion reason");

  const scoreColumns = CONDITIONS.map(([dataset, protocol]) => `${dataset}_${protocol}_raw_score`);
  const identityColumns = ["subject_id", "canonical_finger_position", ...scoreColumns];
  const scoreValues = (identity) => CONDITIONS.map(([dataset, protocol]) => {
    const row = rowMaps.get(`${dataset}/${protocol}`).get(identity.key);
    return row ? row.raw_score : "";
  });
  const includedPath = path.join(OUTPUT_ROOT, "included_identities.csv");
  await writeUtf8(includedPath, csvText(identityColumns, included.map((identity) => [
    identity.subject_id,
    identity.canonical_finger_position,
    ...scoreValues(identity),
  ])));

  const excludedHeaders = [
    "subject_id",
    "canonical_finger_position",
    "reason_flags",
    ...REASONS,
    ...scoreColumns,
  ];
  const excludedPath = path.join(OUTPUT_ROOT, "excluded_identities.csv");
  await writeUtf8(excludedPath, csvText(excludedHeaders, excluded.map((identity) => [
    identity.subject_id,
    identity.canonical_finger_position,
    identity.reasons.join(";"),
    ...REASONS.map((reason) => identity.reasons.includes(reason)),
    ...scoreValues(identity),
  ])));

  const conditionSummaries = [];
  for (const [dataset, protocol] of CONDITIONS) {
    const key = `${dataset}/${protocol}`;
    const table = tables.get(key);
    const filteredRows = table.rows.filter((row) => includedKeys.has(row.key));
    assert(filteredRows.length === included.length, `Filtered row count mismatch in ${key}`);
    const filteredKeys = filteredRows.map((row) => row.key).sort();
    assert(arraysEqual(filteredKeys, [...includedKeys].sort()), `Filtered identity mismatch in ${key}`);
    assert(new Set(filteredRows.map((row) => row.pair_id)).size === filteredRows.length, `Filtered pair_id duplicate in ${key}`);
    assert(filteredRows.every((row) => row.dataset === dataset && row.protocol === protocol), `Filtered dataset/protocol mismatch in ${key}`);
    assert(filteredRows.every((row) => row.status === "ok"), `Non-ok SourceAFIS row in filtered ${key}`);

    const outputPath = path.join(OUTPUT_ROOT, dataset, protocol, "pairs.csv");
    const filteredText = `${table.headerRecord}${table.eol}${filteredRows.map((row) => row.rawRecord).join(table.eol)}${table.eol}`;
    await writeUtf8(outputPath, filteredText);
    const outputRecords = scanCsvRecords(await fs.readFile(outputPath, "utf8"));
    assert(outputRecords[0] === table.headerRecord, `Filtered header changed in ${key}`);
    assert(arraysEqual(outputRecords.slice(1), filteredRows.map((row) => row.rawRecord)), `Filtered source rows changed in ${key}`);

    const rawScores = filteredRows.map((row) => Number(row.raw_score));
    const methodCompare = filteredRows.map((row) => Number(row.method_compare_ms));
    assert(rawScores.every(Number.isFinite), `Invalid raw_score in filtered ${key}`);
    assert(methodCompare.every((value) => Number.isFinite(value) && value >= 0), `Invalid method_compare_ms in filtered ${key}`);
    const zeroScoreCount = rawScores.filter((score) => score === 0).length;
    if (protocol !== "plain_roll") assert(zeroScoreCount === 0, `Self zero score remained in filtered ${key}`);
    conditionSummaries.push({
      dataset,
      protocol,
      filtered_row_count: filteredRows.length,
      zero_score_count: zeroScoreCount,
      mean_raw_score: rounded(mean(rawScores)),
      median_raw_score: rounded(median(rawScores)),
      mean_method_compare_ms: rounded(mean(methodCompare)),
      source_result_path: rel(table.filePath),
      source_result_sha256: beforeHashes[rel(table.filePath)],
      manifest_path: rel(manifests.get(key).filePath),
      manifest_sha256: beforeHashes[rel(manifests.get(key).filePath)],
      filtered_output_path: rel(outputPath),
      filtered_output_sha256: await sha256(outputPath),
    });
  }

  const includedSubjects = [...new Set(included.map((identity) => identity.subject_id))].sort();
  const fingerDistribution = Object.fromEntries(
    Array.from({ length: 10 }, (_, index) => index + 1).map((position) => [
      String(position),
      included.filter((identity) => identity.canonical_finger_position === position).length,
    ]),
  );
  const exclusionCounts = Object.fromEntries(REASONS.map((reason) => [
    reason,
    excluded.filter((identity) => identity.reasons.includes(reason)).length,
  ]));
  const multiReasonCount = excluded.filter((identity) => identity.reasons.length > 1).length;
  const excludedSolelyPlainRollZero = excluded.filter((identity) => {
    const plainRollZero = identity.scores["sd300b/plain_roll"] === 0 || identity.scores["sd300c/plain_roll"] === 0;
    return plainRollZero && identity.reasons.length === 0;
  }).length;
  assert(excludedSolelyPlainRollZero === 0, "Identity was excluded solely because of a plain_roll zero score");

  const summaryCsvHeaders = [
    "cohort_name",
    "cohort_description",
    "dataset",
    "protocol",
    "base_identity_count",
    "included_identity_count",
    "excluded_identity_count",
    "distinct_subject_count",
    "filtered_row_count",
    "zero_score_count",
    "mean_raw_score",
    "median_raw_score",
    "mean_method_compare_ms",
    "source_result_path",
    "source_result_sha256",
    "manifest_path",
    "manifest_sha256",
    "filtered_output_path",
    "filtered_output_sha256",
  ];
  const summaryCsvPath = path.join(OUTPUT_ROOT, "cohort_summary.csv");
  await writeUtf8(summaryCsvPath, csvText(summaryCsvHeaders, conditionSummaries.map((summary) => [
    COHORT_NAME,
    COHORT_DESCRIPTION,
    summary.dataset,
    summary.protocol,
    identities.length,
    included.length,
    excluded.length,
    includedSubjects.length,
    summary.filtered_row_count,
    summary.zero_score_count,
    numberText(summary.mean_raw_score),
    numberText(summary.median_raw_score),
    numberText(summary.mean_method_compare_ms),
    summary.source_result_path,
    summary.source_result_sha256,
    summary.manifest_path,
    summary.manifest_sha256,
    summary.filtered_output_path,
    summary.filtered_output_sha256,
  ])));

  const outputHashes = {
    [rel(includedPath)]: await sha256(includedPath),
    [rel(excludedPath)]: await sha256(excludedPath),
    [rel(summaryCsvPath)]: await sha256(summaryCsvPath),
    ...Object.fromEntries(conditionSummaries.map((summary) => [summary.filtered_output_path, summary.filtered_output_sha256])),
  };
  const afterHashes = Object.fromEntries(await Promise.all(protectedFiles.map(async (filePath) => [rel(filePath), await sha256(filePath)])));
  assert(JSON.stringify(beforeHashes) === JSON.stringify(afterHashes), "A protected source artifact changed during cohort construction");

  const summary = {
    cohort_name: COHORT_NAME,
    cohort_version: COHORT_VERSION,
    cohort_description: COHORT_DESCRIPTION,
    selection_note: "This cohort is selected by SourceAFIS performance and is not a substitute for the full, unfiltered result.",
    identity_unit: ["subject_id", "canonical_finger_position"],
    cohort_rule: COHORT_RULE,
    source_bundle_contract: "pairwise-benchmark-v2",
    source_bundle_config_hash: BUNDLE_ID,
    deterministic_summary: true,
    timestamps_in_deterministic_summaries: false,
    summary_numeric_rounding_decimal_places: 12,
    counts: {
      base_identity_count: identities.length,
      included_identity_count: included.length,
      excluded_identity_count: excluded.length,
      exclusion_count_by_reason_flag: exclusionCounts,
      excluded_identity_count_with_multiple_reasons: multiReasonCount,
      distinct_included_subject_count: includedSubjects.length,
      canonical_finger_distribution: fingerDistribution,
      retained_plain_roll_zero_score_count: {
        sd300b: conditionSummaries.find((summary) => summary.dataset === "sd300b" && summary.protocol === "plain_roll").zero_score_count,
        sd300c: conditionSummaries.find((summary) => summary.dataset === "sd300c" && summary.protocol === "plain_roll").zero_score_count,
      },
    },
    condition_summaries: conditionSummaries,
    provenance: {
      builder_path: rel(SCRIPT_PATH),
      builder_sha256: await sha256(SCRIPT_PATH),
      source_result_files_sha256: Object.fromEntries(sourceFiles.map((filePath) => [rel(filePath), beforeHashes[rel(filePath)]])),
      source_manifest_files_sha256: Object.fromEntries(manifestFiles.map((filePath) => [rel(filePath), beforeHashes[rel(filePath)]])),
      zero_score_audit_artifacts_sha256: Object.fromEntries(auditFiles.map((filePath) => [rel(filePath), beforeHashes[rel(filePath)]])),
      included_identities_sha256: outputHashes[rel(includedPath)],
      filtered_output_files_sha256: Object.fromEntries(conditionSummaries.map((summary) => [summary.filtered_output_path, summary.filtered_output_sha256])),
      other_deterministic_output_files_sha256: Object.fromEntries(Object.entries(outputHashes).filter(([filePath]) => !filePath.endsWith("/pairs.csv") && filePath !== rel(includedPath))),
    },
    validation: {
      plain_roll_base_identity_sets_equal: true,
      all_six_source_results_match_manifests_by_pair_id_and_identity: true,
      all_six_filtered_identity_sets_equal: true,
      all_six_filtered_row_counts_equal: true,
      all_filtered_pair_ids_unique: true,
      all_filtered_dataset_protocol_fields_correct: true,
      filtered_score_and_timing_rows_identical_to_source: true,
      self_protocol_zero_score_count_is_zero: true,
      included_identity_with_nonpositive_self_score_count: 0,
      excluded_solely_because_plain_roll_score_is_zero_count: excludedSolelyPlainRollZero,
      protected_source_artifacts_unchanged: true,
      sourceafis_rerun_performed: false,
      java_sidecar_started: false,
      benchmark_runner_started: false,
      reproducible_from_source_bundles_and_rule_only: true,
    },
  };
  const summaryJsonPath = path.join(OUTPUT_ROOT, "cohort_summary.json");
  await writeUtf8(summaryJsonPath, `${JSON.stringify(summary, null, 2)}\n`);

  process.stdout.write(`${JSON.stringify({
    output_root: rel(OUTPUT_ROOT),
    base_identity_count: identities.length,
    included_identity_count: included.length,
    excluded_identity_count: excluded.length,
    exclusion_count_by_reason_flag: exclusionCounts,
    distinct_included_subject_count: includedSubjects.length,
    canonical_finger_distribution: fingerDistribution,
    condition_summaries: conditionSummaries,
    output_hashes: { ...outputHashes, [rel(summaryJsonPath)]: await sha256(summaryJsonPath) },
  }, null, 2)}\n`);
}

await main();
