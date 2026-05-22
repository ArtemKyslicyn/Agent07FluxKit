//
//  Tokenizer.swift
//  FluxKit
//
//  CLIP BPE tokenizer for text conditioning.
//  T5 tokenizer loaded via swift-transformers AutoTokenizer.
//

import Foundation
import Tokenizers
import Hub

// MARK: - CLIP Tokenizer (BPE)

/// BPE tokenizer for CLIP text encoder.
/// Reads merges.txt and vocab.json from the tokenizer directory.
public class CLIPBPETokenizer: @unchecked Sendable {
    private let vocab: [String: Int]
    private let merges: [(String, String)]
    private let bosToken: Int
    private let eosToken: Int
    private var cache: [String: [Int]] = [:]

    public init(vocabURL: URL, mergesURL: URL) throws {
        let vocabData = try Data(contentsOf: vocabURL)
        guard let v = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] else {
            throw TokenizerError.invalidVocab
        }
        self.vocab = v

        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        self.merges = mergesText
            .components(separatedBy: "\n")
            .dropFirst()  // Skip header line
            .compactMap { line in
                let parts = line.split(separator: " ")
                guard parts.count == 2 else { return nil }
                return (String(parts[0]), String(parts[1]))
            }

        self.bosToken = v["<|startoftext|>"] ?? 49406
        self.eosToken = v["<|endoftext|>"] ?? 49407
    }

    /// Encode text to token IDs with BOS/EOS, padded/truncated to maxLength.
    public func encode(_ text: String, maxLength: Int = 77) -> [Int] {
        let cleaned = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let words = cleaned.split(separator: " ").map { String($0) + "</w>" }

        var tokens: [Int] = [bosToken]
        for word in words {
            let wordTokens = bpe(word)
            tokens.append(contentsOf: wordTokens)
            if tokens.count >= maxLength - 1 { break }
        }
        tokens.append(eosToken)

        // Pad or truncate
        if tokens.count < maxLength {
            tokens.append(contentsOf: Array(repeating: eosToken, count: maxLength - tokens.count))
        } else if tokens.count > maxLength {
            tokens = Array(tokens.prefix(maxLength - 1)) + [eosToken]
        }
        return tokens
    }

    private func bpe(_ token: String) -> [Int] {
        if let cached = cache[token] { return cached }

        var word = Array(token).map { String($0) }
        if word.isEmpty { return [] }

        while word.count > 1 {
            var bestPair: (String, String)?
            var bestRank = Int.max

            for i in 0..<(word.count - 1) {
                let pair = (word[i], word[i + 1])
                if let rank = merges.firstIndex(where: { $0 == pair }) {
                    if rank < bestRank {
                        bestRank = rank
                        bestPair = pair
                    }
                }
            }

            guard let pair = bestPair else { break }
            let merged = pair.0 + pair.1

            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1, word[i] == pair.0, word[i + 1] == pair.1 {
                    newWord.append(merged)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
        }

        let ids = word.compactMap { vocab[$0] ?? vocab["<|endoftext|>"] }
        cache[token] = ids
        return ids
    }
}

enum TokenizerError: LocalizedError {
    case invalidVocab
    case tokenizerNotFound

    var errorDescription: String? {
        switch self {
        case .invalidVocab: return "Invalid vocab.json"
        case .tokenizerNotFound: return "Tokenizer files not found"
        }
    }
}

// MARK: - T5 Tokenizer Loader

/// Load T5 tokenizer using swift-transformers AutoTokenizer.
public func loadT5Tokenizer(from directory: URL) throws -> any Tokenizer {
    let configURL = directory.appendingPathComponent("tokenizer_config.json")
    let dataURL = directory.appendingPathComponent("tokenizer.json")

    guard FileManager.default.fileExists(atPath: configURL.path),
          FileManager.default.fileExists(atPath: dataURL.path) else {
        throw TokenizerError.tokenizerNotFound
    }

    let hubApi = HubApi()
    let tokenizerConfig = try hubApi.configuration(fileURL: configURL)
    let tokenizerData = try hubApi.configuration(fileURL: dataURL)

    return try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
}
