//! Synthetic test for `v_cmp_class_f32` (VOPC 0x088) verifying all 10 IEEE-754
//! binary32 classes: sNaN, qNaN, +/-inf, +/-normal, +/-subnormal, and +/-0.0.

use victory_shader::dxbc::Container;
use victory_shader::emit::{self, HostShader};
use victory_shader::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use victory_shader::{io, ir, resource, structure};

fn build_v_cmp_class_f32_cs() -> HostShader {
    // s[4:5] = descriptor table address (heap index 0), s0 = f32 bits, s1 = class mask.
    // v_cmp_class_f32_e32 vcc_lo, v1, v2 (VOPC 0x088); v_cndmask_b32_e64 v0, 0, 1, vcc_lo.
    let words: [u32; 11] = [
        0xBE80_0000 | (5 << 16) | (0x03 << 8) | 0x80, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | (8 << 6) | (4 >> 1), // s_load_dwordx4 s[8:11], s[4:5], 0
        125 << 25,
        0x7E00_0000 | (1 << 17) | (0x001 << 9), // v_mov_b32 v1, s0
        0x7E00_0000 | (2 << 17) | (0x001 << 9) | 1, // v_mov_b32 v2, s1
        0x7C00_0000 | (0x088 << 17) | (2 << 9) | 257, // v_cmp_class_f32_e32 vcc_lo, v1, v2
        0x35 << 26 | 0x101 << 16,               // v_cndmask_b32_e64 v0, 0, 1, vcc_lo
        0x80 | (0x81 << 9) | (106 << 18),
        0x38 << 26 | 0x1C << 18, // buffer_store_dword v0, off, s[8:11], 0
        0x80 << 24 | (8 >> 2) << 16,
        0xBF81_0000, // s_endpgm
    ];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let c = Container {
        digest: [0; 16],
        version: (1, 0),
        parts: Vec::new(),
    };
    let shader = emit::emit(&c, &native, &p, &st, &io, &res).expect("emit");

    #[cfg(windows)]
    if let Ok(mut v) =
        emit::validator::Validator::load(emit::validator::Validator::default_path().as_deref())
    {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

#[test]
fn v_cmp_class_f32_lowers_and_validates() {
    let shader = build_v_cmp_class_f32_cs();
    assert!(!shader.dxil.is_empty());
}

#[cfg(windows)]
fn run_on_device(
    label: &str,
    warp: bool,
    shader: &HostShader,
    cases: &[(u32, u32, u32)],
) -> Result<(), String> {
    use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, ID3DBlob};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_TYPELESS, DXGI_SAMPLE_DESC};
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter, IDXGIFactory4,
    };
    use windows::core::Interface;

    let err = |e: windows::core::Error| e.message().to_string();
    // SAFETY: D3D12/DXGI objects outlive each command-list execution via fence synchronization.
    unsafe {
        let mut dev: Option<ID3D12Device> = None;
        if warp {
            let factory: IDXGIFactory4 =
                CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(err)?;
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().map_err(err)?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut dev).map_err(err)?;
        } else {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev).map_err(err)?;
        }
        let dev = dev.ok_or_else(|| "no device".to_string())?;
        let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: D3D_SHADER_MODEL_6_6,
        };
        dev.CheckFeatureSupport(
            D3D12_FEATURE_SHADER_MODEL,
            (&raw mut sm).cast(),
            size_of_val(&sm) as u32,
        )
        .map_err(err)?;
        if sm.HighestShaderModel.0 < D3D_SHADER_MODEL_6_6.0 {
            return Err(format!("SM {:#x} < 6.6", sm.HighestShaderModel.0));
        }

        let param = D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: emit::USER_DATA_SPACE,
                    Num32BitValues: shader.user_data_dwords().max(1),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        let rs_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: 1,
                    pParameters: &param,
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED,
                    ..Default::default()
                },
            },
        };
        let mut blob: Option<ID3DBlob> = None;
        let mut rs_err: Option<ID3DBlob> = None;
        D3D12SerializeVersionedRootSignature(&rs_desc, &mut blob, Some(&mut rs_err))
            .map_err(err)?;
        let blob = blob.ok_or_else(|| "no root signature blob".to_string())?;
        let rs_bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        let root: ID3D12RootSignature = dev.CreateRootSignature(0, rs_bytes).map_err(err)?;
        let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: shader.dxil.as_ptr().cast(),
                BytecodeLength: shader.dxil.len(),
            },
            ..Default::default()
        };
        let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&pso_desc).map_err(err)?;

        let queue: ID3D12CommandQueue = dev
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .expect("CreateCommandQueue");
        let alloc: ID3D12CommandAllocator = dev
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .expect("CreateCommandAllocator");
        let list: ID3D12GraphicsCommandList = dev
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso)
            .expect("CreateCommandList");
        list.Close().expect("Close");

        let make_buf = |heap, flags, state| {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 4,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let mut r: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: heap,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                None,
                &mut r,
            )
            .expect("CreateCommittedResource");
            r.expect("resource")
        };
        let uav = make_buf(
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let rb = make_buf(
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let heap: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
            .expect("CreateDescriptorHeap");
        let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: 1,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        dev.CreateUnorderedAccessView(
            &uav,
            None,
            Some(&uav_desc),
            heap.GetCPUDescriptorHandleForHeapStart(),
        );
        let fence: ID3D12Fence = dev
            .CreateFence(0, D3D12_FENCE_FLAG_NONE)
            .expect("CreateFence");

        for (step, &(src0, mask, want)) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot, u.dword) {
                    (RootType::Constants, 0, 0) => src0,
                    (RootType::Constants, 0, 1) => mask,
                    (RootType::Constants, 0, _) | (RootType::Table, 1, _) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            alloc.Reset().expect("Reset");
            list.Reset(&alloc, &pso).expect("Reset");
            list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.SetComputeRootSignature(&root);
            list.SetComputeRoot32BitConstants(0, ud.len() as u32, ud.as_ptr().cast(), 0);
            list.Dispatch(1, 1, 1);
            let to_copy = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: std::mem::transmute_copy(&uav),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                    }),
                },
            };
            list.ResourceBarrier(&[to_copy]);
            list.CopyBufferRegion(&rb, 0, &uav, 0, 4);
            let to_uav = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: std::mem::transmute_copy(&uav),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_COPY_SOURCE,
                        StateAfter: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    }),
                },
            };
            list.ResourceBarrier(&[to_uav]);
            list.Close().expect("Close");
            queue.ExecuteCommandLists(&[Some(list.cast().expect("cast"))]);
            let val_fence = (step + 1) as u64;
            queue.Signal(&fence, val_fence).expect("Signal");
            while fence.GetCompletedValue() < val_fence {
                dev.GetDeviceRemovedReason().expect("device removed");
                std::thread::yield_now();
            }
            let mut p = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut p)).expect("Map");
            let got = *p.cast::<u32>();
            rb.Unmap(0, None);
            assert_eq!(
                got, want,
                "{label}: v_cmp_class_f32({src0:#010x}, {mask:#05x}) = {got}, want {want}"
            );
        }
        Ok(())
    }
}

#[cfg(windows)]
#[test]
fn v_cmp_class_f32_matches_rdna2() {
    let shader = build_v_cmp_class_f32_cs();

    // All 10 IEEE-754 binary32 classes: bit k of src1 tests class k.
    let classes: &[(u32, u32)] = &[
        (0x7F80_0001, 0), // +sNaN (quiet bit 0x0040_0000 clear)
        (0xFFBF_FFFF, 0), // -sNaN (sign ignored for NaNs)
        (0x7FC0_0000, 1), // +qNaN
        (0xFFFF_FFFF, 1), // -qNaN
        (0xFF80_0000, 2), // -inf
        (0xBF80_0000, 3), // -normal (-1.0)
        (0x8080_0000, 3), // -normal (-min normal)
        (0xFF7F_FFFF, 3), // -normal (-max normal)
        (0x8000_0001, 4), // -subnormal (-min)
        (0x807F_FFFF, 4), // -subnormal (-max)
        (0x8000_0000, 5), // -0.0
        (0x0000_0000, 6), // +0.0
        (0x0000_0001, 7), // +subnormal (+min)
        (0x007F_FFFF, 7), // +subnormal (+max)
        (0x3F80_0000, 8), // +normal (+1.0)
        (0x0080_0000, 8), // +normal (+min normal)
        (0x7F7F_FFFF, 8), // +normal (+max normal)
        (0x7F80_0000, 9), // +inf
    ];
    let mut cases: Vec<(u32, u32, u32)> = Vec::new();
    for &(bits, cls) in classes {
        cases.push((bits, 1u32 << cls, 1));
        cases.push((bits, 0x3FF & !(1u32 << cls), 0));
    }
    cases.extend_from_slice(&[
        (0x7F80_0001, 0x003, 1),       // any NaN: +sNaN
        (0xFFFF_FFFF, 0x003, 1),       // any NaN: -qNaN
        (0x7F80_0000, 0x003, 0),       // any NaN rejects +inf
        (0xFF80_0000, 0x204, 1),       // +/-inf: -inf
        (0x7F80_0000, 0x204, 1),       // +/-inf: +inf
        (0xFF7F_FFFF, 0x204, 0),       // +/-inf rejects -max normal
        (0x8000_0000, 0x060, 1),       // +/-0.0: -0.0
        (0x0000_0000, 0x060, 1),       // +/-0.0: +0.0
        (0x0000_0001, 0x060, 0),       // +/-0.0 rejects +min subnormal
        (0x3F80_0000, 0x000, 0),       // empty mask
        (0x3F80_0000, 0xFFFF_FC00, 0), // unused upper mask bits ignored
    ]);

    let force_warp = std::env::var_os("VICTORY_D3D12_WARP").is_some()
        || std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
    let backends: &[(&str, bool)] = if force_warp {
        &[("warp", true)]
    } else {
        &[("hardware", false), ("warp", true)]
    };

    let mut ran = 0;
    for &(label, warp) in backends {
        if let Err(reason) = run_on_device(label, warp, &shader, &cases) {
            eprintln!("no SM 6.6 {label} device ({reason}); skipped");
            continue;
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}
