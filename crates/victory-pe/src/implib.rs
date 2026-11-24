//! COFF import libraries (`.lib`).
//!
//! An umbrella import library tells us which DLL provides each API that a
//! console title may link against. That is the map the runtime needs to
//! decide which of our implementations answers which import.

use object::read::archive::ArchiveFile;
use object::read::coff::{ImportFile, ImportName, ImportType};

use crate::{ImportSymbol, PeError};

/// One `dll!export` pair described by an import library.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct LibImport {
    pub dll: String,
    pub export: ImportSymbol,
    pub kind: LibImportKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LibImportKind {
    Code,
    Data,
    Const,
}

/// Result of reading an import library.
#[derive(Debug, Default)]
pub struct ImportLibrary {
    pub imports: Vec<LibImport>,
    /// Members that are not short-format import objects (regular objects,
    /// long-format import descriptors). Reported so nothing is silently lost.
    pub other_members: usize,
}

impl ImportLibrary {
    pub fn parse(data: &[u8]) -> Result<Self, PeError> {
        let archive = ArchiveFile::parse(data)?;
        let mut lib = ImportLibrary::default();
        for member in archive.members() {
            let member = member?;
            let bytes = member.data(data)?;
            match ImportFile::parse(bytes) {
                Ok(import) => lib.imports.push(LibImport {
                    dll: String::from_utf8_lossy(import.dll()).into_owned(),
                    export: match import.import() {
                        ImportName::Name(name) => ImportSymbol::Name {
                            hint: 0,
                            name: String::from_utf8_lossy(name).into_owned(),
                        },
                        ImportName::Ordinal(o) => ImportSymbol::Ordinal(o),
                    },
                    kind: match import.import_type() {
                        ImportType::Code => LibImportKind::Code,
                        ImportType::Data => LibImportKind::Data,
                        ImportType::Const => LibImportKind::Const,
                    },
                }),
                Err(_) => lib.other_members += 1,
            }
        }
        lib.imports.sort();
        lib.imports.dedup();
        Ok(lib)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_archive(members: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = b"!<arch>\n".to_vec();
        for (name, data) in members {
            let header = format!(
                "{:<16}{:<12}{:<6}{:<6}{:<8}{:<10}`\n",
                name,
                "0",
                "0",
                "0",
                "100644",
                data.len()
            );
            assert_eq!(header.len(), 60);
            out.extend_from_slice(header.as_bytes());
            out.extend_from_slice(data);
            if data.len() % 2 == 1 {
                out.push(b'\n');
            }
        }
        out
    }

    fn short_import(
        dll: &str,
        symbol: &str,
        ordinal_or_hint: u16,
        import_type: u16,
        name_type: u16,
    ) -> Vec<u8> {
        let size_of_data = (symbol.len() + 1 + dll.len() + 1) as u32;
        let flags = (name_type << 2) | import_type;
        let mut buf = Vec::with_capacity(20 + size_of_data as usize);
        buf.extend_from_slice(&0x0000u16.to_le_bytes()); // sig1
        buf.extend_from_slice(&0xFFFFu16.to_le_bytes()); // sig2
        buf.extend_from_slice(&0u16.to_le_bytes()); // version
        buf.extend_from_slice(&0x8664u16.to_le_bytes()); // machine (AMD64)
        buf.extend_from_slice(&0u32.to_le_bytes()); // time_date_stamp
        buf.extend_from_slice(&size_of_data.to_le_bytes());
        buf.extend_from_slice(&ordinal_or_hint.to_le_bytes());
        buf.extend_from_slice(&flags.to_le_bytes());
        buf.extend_from_slice(symbol.as_bytes());
        buf.push(0);
        buf.extend_from_slice(dll.as_bytes());
        buf.push(0);
        buf
    }

    #[test]
    fn parses_name_imports_all_kinds_sorted_and_deduped() {
        let m1 = short_import("b.dll", "DataSym", 3, 1, 1); // Data, Name
        let m2 = short_import("a.dll", "ConstSym", 7, 2, 1); // Const, Name
        let m3 = short_import("a.dll", "CodeSym", 1, 0, 1); // Code, Name
        let m4 = short_import("a.dll", "CodeSym", 9, 0, 1); // Duplicate after hint=0 normalization
        let archive = build_archive(&[
            ("b.dll/", &m1),
            ("a.dll/", &m2),
            ("a.dll/", &m3),
            ("a.dll/", &m4),
        ]);

        let lib = ImportLibrary::parse(&archive).unwrap();
        assert_eq!(lib.other_members, 0);
        assert_eq!(
            lib.imports,
            vec![
                LibImport {
                    dll: "a.dll".into(),
                    export: ImportSymbol::Name {
                        hint: 0,
                        name: "CodeSym".into()
                    },
                    kind: LibImportKind::Code,
                },
                LibImport {
                    dll: "a.dll".into(),
                    export: ImportSymbol::Name {
                        hint: 0,
                        name: "ConstSym".into()
                    },
                    kind: LibImportKind::Const,
                },
                LibImport {
                    dll: "b.dll".into(),
                    export: ImportSymbol::Name {
                        hint: 0,
                        name: "DataSym".into()
                    },
                    kind: LibImportKind::Data,
                },
            ]
        );
    }

    #[test]
    fn parses_ordinal_imports() {
        let ord_code = short_import("xapi.dll", "IgnoredName", 42, 0, 0);
        let ord_data = short_import("xapi.dll", "IgnoredData", 99, 1, 0);
        let archive = build_archive(&[("xapi.dll/", &ord_code), ("xapi.dll/", &ord_data)]);

        let lib = ImportLibrary::parse(&archive).unwrap();
        assert_eq!(lib.other_members, 0);
        assert_eq!(
            lib.imports,
            vec![
                LibImport {
                    dll: "xapi.dll".into(),
                    export: ImportSymbol::Ordinal(42),
                    kind: LibImportKind::Code,
                },
                LibImport {
                    dll: "xapi.dll".into(),
                    export: ImportSymbol::Ordinal(99),
                    kind: LibImportKind::Data,
                },
            ]
        );
    }

    #[test]
    fn counts_non_import_members_as_other() {
        let valid = short_import("kernel.dll", "KeSleep", 0, 0, 1);
        let archive = build_archive(&[("obj.o/", b"not-a-short-import"), ("kernel.dll/", &valid)]);

        let lib = ImportLibrary::parse(&archive).unwrap();
        assert_eq!(lib.other_members, 1);
        assert_eq!(lib.imports.len(), 1);
        assert_eq!(lib.imports[0].dll, "kernel.dll");
    }

    #[test]
    fn rejects_malformed_archives() {
        assert!(ImportLibrary::parse(b"").is_err());
        assert!(ImportLibrary::parse(b"not-an-archive\n").is_err());

        let truncated_header = b"!<arch>\nfoo.dll/";
        assert!(ImportLibrary::parse(truncated_header).is_err());

        let mut truncated_member = b"!<arch>\n".to_vec();
        let hdr = format!(
            "{:<16}{:<12}{:<6}{:<6}{:<8}{:<10}`\n",
            "a/", "0", "0", "0", "644", "100"
        );
        truncated_member.extend_from_slice(hdr.as_bytes());
        truncated_member.extend_from_slice(b"short");
        assert!(ImportLibrary::parse(&truncated_member).is_err());
    }
}
