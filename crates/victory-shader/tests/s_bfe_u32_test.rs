//! Synthetic scalar test for `s_bfe_u32` (SOP2 0x027) covering zero-extension
//! and 7-bit width / 5-bit offset masking.

use victory_shader::dxbc::Container;
use victory_shader::emit::{self, HostShader};
use victory_shader::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use victory_shader::{io, ir, resource, structure};

const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

fn build_s_bfe_u32_cs() -> HostShader {
    let words: [u32; 8] = [
        0xBE80_0000 | (5 << 16) | (0x03 << 8) | 0x80, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | (8 << 6) | (4 >> 1), // s_load_dwordx4 s[8:11], s[4:5], 0
        125 << 25,
        sop2(0x27, 6, 0, 1),            // s_bfe_u32 s6, s0, s1
        0x7E00_0000 | (0x001 << 9) | 6, // v_mov_b32 v0, s6
        0x38 << 26 | 0x1C << 18,        // buffer_store_dword v0, off, s[8:11], 0
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
fn s_bfe_u32_lowers_and_validates() {
    let shader = build_s_bfe_u32_cs();
    assert!(!shader.dxil.is_empty());
}

#[cfg(all(windows, feature = "host"))]
fn dispatch_one_dword(
    dev: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
    root: &windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
    pso: &windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
    ud: &[u32],
) -> u32 {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_TYPELESS, DXGI_SAMPLE_DESC};
    use windows::core::Interface;

    // SAFETY: D3D12 objects outlive command-list execution via fence wait.
    unsafe {
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
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, pso)
            .expect("CreateCommandList");
        let buf_desc = |flags| D3D12_RESOURCE_DESC {
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
        let make = |heap, flags, state| {
            let mut r: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: heap,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &buf_desc(flags),
                state,
                None,
                &mut r,
            )
            .expect("CreateCommittedResource");
            r.expect("resource")
        };
        let uav = make(
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        let rb = make(
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
        list.SetDescriptorHeaps(&[Some(heap.clone())]);
        list.SetComputeRootSignature(root);
        list.SetComputeRoot32BitConstants(0, ud.len() as u32, ud.as_ptr().cast(), 0);
        list.Dispatch(1, 1, 1);
        let barrier = D3D12_RESOURCE_BARRIER {
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
        list.ResourceBarrier(&[barrier]);
        list.CopyBufferRegion(&rb, 0, &uav, 0, 4);
        list.Close().expect("Close");
        queue.ExecuteCommandLists(&[Some(list.cast().expect("cast"))]);
        let fence: ID3D12Fence = dev
            .CreateFence(0, D3D12_FENCE_FLAG_NONE)
            .expect("CreateFence");
        queue.Signal(&fence, 1).expect("Signal");
        while fence.GetCompletedValue() < 1 {
            dev.GetDeviceRemovedReason().expect("device removed");
            std::thread::yield_now();
        }
        let mut p = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut p)).expect("Map");
        let val = *p.cast::<u32>();
        rb.Unmap(0, None);
        val
    }
}

#[cfg(all(windows, feature = "host"))]
#[test]
fn s_bfe_u32_matches_rdna2() {
    use victory_shader::emit::host::Host;

    let shader = build_s_bfe_u32_cs();
    let cases: &[(u32, u32, u32)] = &[
        (0xD2F9_0159, (16 << 16) | 28, 0x0000_000D),
        (0x8000_0000, (4 << 16) | 31, 0x0000_0001),
        (0x8000_0000, (31 << 16) | 4, 0x0800_0000),
        (0x0000_0080, (4 << 16) | 4, 0x0000_0008),
        (0x0000_0070, (4 << 16) | 4, 0x0000_0007),
        (0xFFFF_FFFF, 4, 0x0000_0000),
        (0x8000_0000, (32 << 16) | 4, 0x0800_0000),
        (0x8000_0000, 32 << 16, 0x8000_0000),
        (0xDEAD_BEEF, (64 << 16) | 8, 0x00DE_ADBE),
        (0xDEAD_BEEF, (127 << 16) | 4, 0x0DEA_DBEE),
        (0xFFFF_FFFF, (1 << 23) | 4, 0x0000_0000),
        (0x1234_5678, 0xFF80_0000 | (4 << 16) | 4, 0x0000_0007),
        (0x1234_5678, (8 << 16) | 0xFFE4, 0x0000_0067),
        (0x0000_00F0, (4 << 16) | 36, 0x0000_000F),
    ];

    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for &(src0, src1, want) in cases {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot, u.dword) {
                    (RootType::Constants, 0, 0) => src0,
                    (RootType::Constants, 0, 1) => src1,
                    (RootType::Constants, 0, _) | (RootType::Table, 1, _) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let got = dispatch_one_dword(&host.device, &root, &pso, &ud);
            assert_eq!(
                got, want,
                "{label}: s_bfe_u32({src0:#010x}, {src1:#010x}) = {got:#010x}, want {want:#010x}"
            );
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}
