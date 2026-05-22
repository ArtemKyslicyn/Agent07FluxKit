//
//  FluxKit.swift
//  FluxKit
//
//  MIT-licensed Flux image generation on Apple Silicon via MLX.
//  Clean-room implementation based on published Flux architecture.
//
//  Copyright (c) 2025-2026 Artem Kislitsyn. MIT License.
//

@_exported import MLX

// Public API re-exported from submodules:
// - FluxConfiguration, LoadConfiguration, EvaluateParameters  (Configuration.swift)
// - TextToImageGenerator, DenoiseIterator                     (Pipeline.swift)
// - MultiModalDiffusionTransformer                            (Transformer.swift)
// - T5Encoder, CLIPEncoder                                    (TextEncoders.swift)
// - VAE                                                       (VAE.swift)
