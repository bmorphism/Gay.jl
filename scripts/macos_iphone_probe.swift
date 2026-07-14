#!/usr/bin/swift

import AppKit
import ApplicationServices
import Foundation

private let schema = "gay-iphone-probe/v1"
private let maximumCoreDeviceBytes = 4 * 1024 * 1024

private struct Observation: Codable {
    let schema: String
    let complete: Bool
    let state: String?
    let voice_memos_sync: Bool?
    let recording_count_bin: Int?
    let interaction_bin: Int?
    let connection_evidence: String
    let sync_evidence: String
    let recordings_evidence: String
}

private func attribute(_ element: AXUIElement, _ name: String) -> AnyObject? {
    var value: CFTypeRef?
    guard AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success else {
        return nil
    }
    return value as AnyObject?
}

private func children(_ element: AXUIElement) -> [AXUIElement] {
    attribute(element, kAXChildrenAttribute as String) as? [AXUIElement] ?? []
}

private func applicationRoot(bundleIdentifier: String) -> AXUIElement? {
    guard let application = NSRunningApplication
        .runningApplications(withBundleIdentifier: bundleIdentifier).first else {
        return nil
    }
    return AXUIElementCreateApplication(application.processIdentifier)
}

private func walk(_ root: AXUIElement, maximumDepth: Int = 12,
                  maximumNodes: Int = 4096,
                  visit: (AXUIElement) -> Void) {
    var remaining = maximumNodes
    func descend(_ element: AXUIElement, _ depth: Int) {
        guard depth <= maximumDepth, remaining > 0 else { return }
        remaining -= 1
        visit(element)
        for child in children(element) {
            descend(child, depth + 1)
        }
    }
    descend(root, 0)
}

private func textAttribute(_ element: AXUIElement, _ name: String) -> String? {
    attribute(element, name) as? String
}

private func connection(from label: String)
    -> (state: String?, interaction: Int?, evidence: String, priority: Int) {
    let normalized = label.lowercased()
    if normalized == "connection paused" || normalized == "resume" {
        return ("available", 2, "ax-connection-paused", 2)
    }
    if normalized == "connection interrupted" {
        return ("interrupted", 1, "ax-connection-interrupted", 3)
    }
    if normalized == "iphone mirroring is locked" {
        return ("interrupted", 1, "ax-local-auth-required", 3)
    }
    if normalized == "iphone in use" || normalized == "lock your iphone to connect" {
        return ("interrupted", 1, "ax-remote-control-gated", 3)
    }
    if normalized == "iphone not available" {
        return ("unavailable", 0, "ax-iphone-unavailable", 4)
    }
    if normalized == "connecting to iphone" {
        return ("available", 1, "ax-connecting", 1)
    }
    return (nil, nil, "ax-status-unknown", 0)
}

private func observeAXConnection() -> (String?, Int?, String) {
    guard let root = applicationRoot(bundleIdentifier: "com.apple.ScreenContinuity"),
          let windows = attribute(root, kAXWindowsAttribute as String) as? [AXUIElement],
          !windows.isEmpty else {
        return (nil, nil, "ax-window-unavailable")
    }
    var result: (state: String?, interaction: Int?, evidence: String, priority: Int) =
        (nil, nil, "ax-status-unknown", 0)
    for window in windows {
        // Status overlays live in the shallow window shell. Do not descend
        // into a mirrored iPhone app's accessibility content.
        walk(window, maximumDepth: 4, maximumNodes: 64) { element in
            for name in [kAXTitleAttribute, kAXDescriptionAttribute, kAXValueAttribute] {
                if let value = textAttribute(element, name as String), !value.isEmpty {
                    let candidate = connection(from: value)
                    if candidate.priority > result.priority {
                        result = candidate
                    }
                }
            }
        }
    }
    return (result.state, result.interaction, result.evidence)
}

private func coreDeviceConnection(deviceCount: Int,
                                  pairingState: String?,
                                  tunnelState: String?)
    -> (state: String?, interaction: Int?, evidence: String) {
    if deviceCount == 0 { return (nil, nil, "coredevice-none") }
    if deviceCount > 1 { return (nil, nil, "coredevice-ambiguous") }
    guard let pairingState else {
        return (nil, nil, "coredevice-status-unknown")
    }
    if pairingState == "unpaired" {
        return ("unavailable", 0, "coredevice-unpaired")
    }
    guard pairingState == "paired" else {
        return (nil, nil, "coredevice-status-unknown")
    }
    switch tunnelState {
    case "connected": return ("connected", 3, "coredevice-connected")
    case "available": return ("available", 2, "coredevice-available")
    case "unavailable": return ("unavailable", 0, "coredevice-unavailable")
    default: return (nil, nil, "coredevice-status-unknown")
    }
}

private func coreDeviceConnection(from data: Data) ->
    (state: String?, interaction: Int?, evidence: String) {
    guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let result = root["result"] as? [String: Any],
          let devices = result["devices"] as? [[String: Any]] else {
        return (nil, nil, "coredevice-probe-unavailable")
    }
    guard devices.count == 1 else {
        return coreDeviceConnection(deviceCount: devices.count,
                                    pairingState: nil, tunnelState: nil)
    }
    let properties = devices[0]["connectionProperties"] as? [String: Any]
    return coreDeviceConnection(
        deviceCount: 1,
        pairingState: properties?["pairingState"] as? String,
        tunnelState: properties?["tunnelState"] as? String)
}

private func readBounded(_ handle: FileHandle, maximumBytes: Int) throws -> Data? {
    precondition(maximumBytes >= 0)
    var result = Data()
    while true {
        let remaining = maximumBytes - result.count
        let chunk = try handle.read(upToCount: min(64 * 1024, remaining + 1)) ?? Data()
        if chunk.isEmpty { return result }
        guard chunk.count <= remaining else { return nil }
        result.append(chunk)
    }
}

private func observeCoreDeviceConnection() -> (String?, Int?, String) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
    process.arguments = [
        "devicectl", "list", "devices",
        "--filter", "hardwareProperties.deviceType == 'iPhone'",
        "--quiet", "--timeout", "5", "--json-output", "/dev/stdout",
    ]
    let stdout = Pipe()
    process.standardOutput = stdout
    process.standardError = FileHandle.nullDevice
    do {
        try process.run()
        guard let data = try readBounded(stdout.fileHandleForReading,
                                         maximumBytes: maximumCoreDeviceBytes) else {
            if process.isRunning { process.terminate() }
            process.waitUntilExit()
            return (nil, nil, "coredevice-probe-unavailable")
        }
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            return (nil, nil, "coredevice-probe-unavailable")
        }
        return coreDeviceConnection(from: data)
    } catch {
        if process.isRunning {
            process.terminate()
            process.waitUntilExit()
        }
        return (nil, nil, "coredevice-probe-unavailable")
    }
}

private func observeConnection() -> (String?, Int?, String) {
    let accessibility = observeAXConnection()
    if accessibility.0 != nil { return accessibility }
    return observeCoreDeviceConnection()
}

private func countBin(_ count: Int) -> Int {
    count == 0 ? 0 : count <= 4 ? 1 : count <= 16 ? 2 : 3
}

private func recordings(from label: String) -> (Int?, String) {
    let expression = try! NSRegularExpression(
        pattern: #"^All Recordings, ([0-9][0-9,]*) recordings?$"#)
    let range = NSRange(label.startIndex..<label.endIndex, in: label)
    guard let match = expression.firstMatch(in: label, range: range),
          let capture = Range(match.range(at: 1), in: label) else {
        return (nil, "ax-all-recordings-unavailable")
    }
    let digits = label[capture].replacingOccurrences(of: ",", with: "")
    guard let exactCount = Int(digits), exactCount >= 0 else {
        return (nil, "ax-all-recordings-unavailable")
    }
    // Coarsen immediately. The exact count is not persisted or emitted.
    return (countBin(exactCount), "ax-selected-all-recordings")
}

private func observeRecordings() -> (Int?, String) {
    guard let root = applicationRoot(bundleIdentifier: "com.apple.VoiceMemos"),
          let windows = attribute(root, kAXWindowsAttribute as String) as? [AXUIElement],
          !windows.isEmpty else {
        return (nil, "ax-window-unavailable")
    }
    guard let allRecordingsWindow = windows.first(where: { window in
        textAttribute(window, kAXTitleAttribute as String) == "All Recordings"
    }) else {
        return (nil, "ax-all-recordings-not-selected")
    }

    var foldersList: AXUIElement?
    walk(allRecordingsWindow, maximumDepth: 10, maximumNodes: 256) { element in
        guard foldersList == nil else { return }
        if textAttribute(element, kAXIdentifierAttribute as String) == "FoldersList" {
            foldersList = element
        }
    }
    guard let foldersList else {
        return (nil, "ax-all-recordings-unavailable")
    }

    var result: (Int?, String) = (nil, "ax-all-recordings-unavailable")
    // Scan only buttons in FoldersList. Test each attribute on the fly and
    // retain only the coarse bin; recording-card titles are never read.
    walk(foldersList, maximumDepth: 4, maximumNodes: 64) { element in
        guard result.0 == nil else { return }
        guard textAttribute(element, kAXRoleAttribute as String) == (kAXButtonRole as String)
        else { return }
        for name in [kAXTitleAttribute, kAXValueAttribute,
                     kAXDescriptionAttribute, kAXHelpAttribute] {
            guard let value = textAttribute(element, name as String) else { continue }
            let candidate = recordings(from: value)
            if candidate.0 != nil { result = candidate }
        }
    }
    return result
}

private func observeSync() -> (Bool?, String) {
    let bundleIdentifiers = ["com.apple.systempreferences", "com.apple.SystemSettings"]
    for bundleIdentifier in bundleIdentifiers {
        guard let root = applicationRoot(bundleIdentifier: bundleIdentifier),
              let windows = attribute(root, kAXWindowsAttribute as String) as? [AXUIElement],
              !windows.isEmpty else { continue }
        var result: Bool?
        for window in windows {
            walk(window) { element in
                guard result == nil else { return }
                let identifier = textAttribute(element, kAXIdentifierAttribute as String) ?? ""
                let description = textAttribute(element, kAXDescriptionAttribute as String) ?? ""
                guard identifier == "toggle-Voice-Memos" ||
                      description == "Toggle for Voice Memos" else { return }
                if let number = attribute(element, kAXValueAttribute as String) as? NSNumber {
                    result = number.boolValue
                }
            }
            if result != nil { break }
        }
        if let result { return (result, "ax-icloud-voice-memos-toggle") }
    }
    return (nil, "ax-toggle-unavailable")
}

private func makeObservation() -> Observation {
    let connectionResult = observeConnection()
    let syncResult = observeSync()
    let recordingsResult = observeRecordings()
    let complete = connectionResult.0 != nil && connectionResult.1 != nil &&
                   syncResult.0 != nil && recordingsResult.0 != nil
    return Observation(
        schema: schema,
        complete: complete,
        state: connectionResult.0,
        voice_memos_sync: syncResult.0,
        recording_count_bin: recordingsResult.0,
        interaction_bin: connectionResult.1,
        connection_evidence: connectionResult.2,
        sync_evidence: syncResult.1,
        recordings_evidence: recordingsResult.1)
}

private func encodedJSON(_ observation: Observation) throws -> Data {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    return try encoder.encode(observation)
}

private func emitJSON(_ observation: Observation) throws {
    let data = try encodedJSON(observation)
    print(String(decoding: data, as: UTF8.self))
}

private func field(_ value: String?) -> String { value ?? "-" }
private func field(_ value: Bool?) -> String { value.map { $0 ? "1" : "0" } ?? "-" }
private func field(_ value: Int?) -> String { value.map(String.init) ?? "-" }

private func emitTSV(_ observation: Observation) {
    print([
        observation.schema,
        field(observation.state),
        field(observation.voice_memos_sync),
        field(observation.recording_count_bin),
        field(observation.interaction_bin),
        observation.connection_evidence,
        observation.sync_evidence,
        observation.recordings_evidence,
    ].joined(separator: "\t"))
}

private func selfTest() {
    precondition(connection(from: "Connection Paused").state == "available")
    precondition(connection(from: "Connection Interrupted").state == "interrupted")
    precondition(connection(from: "iPhone Mirroring Is Locked").state == "interrupted")
    let precedence = ["Resume", "Connection Interrupted", "iPhone Not Available"]
        .map(connection(from:)).max { $0.priority < $1.priority }
    precondition(precedence?.state == "unavailable")
    precondition(coreDeviceConnection(deviceCount: 0, pairingState: nil,
                                      tunnelState: nil).state == nil)
    precondition(coreDeviceConnection(deviceCount: 2, pairingState: "paired",
                                      tunnelState: "connected").state == nil)
    precondition(coreDeviceConnection(deviceCount: 1, pairingState: "paired",
                                      tunnelState: "connected").state == "connected")
    precondition(coreDeviceConnection(deviceCount: 1, pairingState: "future-state",
                                      tunnelState: "connected").state == nil)
    let coreDeviceFixture = Data(#"{"result":{"devices":[{"name":"forbidden-device-name","identifier":"forbidden-device-id","connectionProperties":{"pairingState":"paired","tunnelState":"available"}}]}}"#.utf8)
    let coreDeviceResult = coreDeviceConnection(from: coreDeviceFixture)
    precondition(coreDeviceResult.state == "available")
    precondition(coreDeviceResult.interaction == 2)
    do {
        let bounded = Pipe()
        bounded.fileHandleForWriting.write(Data([1, 2, 3, 4]))
        bounded.fileHandleForWriting.closeFile()
        precondition(try! readBounded(bounded.fileHandleForReading,
                                     maximumBytes: 4) == Data([1, 2, 3, 4]))
        let oversized = Pipe()
        oversized.fileHandleForWriting.write(Data([1, 2, 3, 4, 5]))
        oversized.fileHandleForWriting.closeFile()
        precondition(try! readBounded(oversized.fileHandleForReading,
                                     maximumBytes: 4) == nil)
    }
    precondition(countBin(0) == 0 && countBin(4) == 1 &&
                 countBin(16) == 2 && countBin(17) == 3)
    let recordingsResult = recordings(from: "All Recordings, 17 recordings")
    precondition(recordingsResult.0 == 3)
    precondition(recordingsResult.1 == "ax-selected-all-recordings")
    let safe = Observation(
        schema: schema, complete: true, state: "available",
        voice_memos_sync: true, recording_count_bin: recordingsResult.0,
        interaction_bin: coreDeviceResult.interaction,
        connection_evidence: coreDeviceResult.evidence,
        sync_evidence: "ax-icloud-voice-memos-toggle",
        recordings_evidence: recordingsResult.1)
    let encoded = String(decoding: try! encodedJSON(safe), as: UTF8.self)
    precondition(!encoded.contains("17 recordings"))
    precondition(!encoded.contains("recording-title"))
    precondition(!encoded.contains("forbidden-device-name"))
    precondition(!encoded.contains("forbidden-device-id"))
    print("ok")
}

let arguments = Array(CommandLine.arguments.dropFirst())
if arguments.contains("--self-test") {
    selfTest()
} else {
    let formatIndex = arguments.firstIndex(of: "--format")
    let format = formatIndex.flatMap { index in
        arguments.indices.contains(index + 1) ? arguments[index + 1] : nil
    } ?? "json"
    let observation = makeObservation()
    if format == "tsv" {
        emitTSV(observation)
    } else if format == "json" {
        try emitJSON(observation)
    } else {
        fputs("unsupported format\n", stderr)
        exit(64)
    }
}
