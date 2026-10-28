// Stroke icons from the design mockups.
type Props = { size?: number; color?: string };

const base = (size: number, color: string, width = 2) => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: color,
  strokeWidth: width,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
});

export const LogoIcon = ({ size = 26, color = "#2B54D9" }: Props) => (
  <svg {...base(size, color)}>
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <path d="M14 2v6h6" />
    <circle cx="11" cy="14" r="3" />
    <path d="m13.5 16.5 2.5 2.5" />
  </svg>
);

export const PlusIcon = ({ size = 16, color = "currentColor" }: Props) => (
  <svg {...base(size, color, 2.2)}>
    <path d="M12 5v14M5 12h14" />
  </svg>
);

export const SendIcon = ({ size = 18, color = "currentColor" }: Props) => (
  <svg {...base(size, color)}>
    <path d="M22 2 11 13" />
    <path d="M22 2 15 22l-4-9-9-4z" />
  </svg>
);

export const UploadIcon = ({ size = 22, color = "#2B54D9" }: Props) => (
  <svg {...base(size, color)}>
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <path d="m17 8-5-5-5 5" />
    <path d="M12 3v12" />
  </svg>
);

export const HistoryIcon = ({ size = 13, color = "currentColor" }: Props) => (
  <svg {...base(size, color)}>
    <path d="M21 12a9 9 0 1 1-3-6.7L21 8" />
    <path d="M21 3v5h-5" />
  </svg>
);

export const AlertIcon = ({ size = 18, color = "#8A4B00" }: Props) => (
  <svg {...base(size, color)} style={{ flex: "0 0 auto", marginTop: 2 }}>
    <circle cx="12" cy="12" r="10" />
    <path d="M12 8v4M12 16h.01" />
  </svg>
);

export const InfoIcon = ({ size = 18, color = "#1C3BA8" }: Props) => (
  <svg {...base(size, color)} style={{ flex: "0 0 auto" }}>
    <circle cx="12" cy="12" r="10" />
    <path d="M12 16v-4M12 8h.01" />
  </svg>
);

export const CloseIcon = ({ size = 16, color = "currentColor" }: Props) => (
  <svg {...base(size, color)}>
    <path d="M18 6 6 18M6 6l12 12" />
  </svg>
);

export const TrashIcon = ({ size = 15, color = "currentColor" }: Props) => (
  <svg {...base(size, color)}>
    <path d="M3 6h18" />
    <path d="M8 6V4h8v2" />
    <path d="M19 6l-1 14H6L5 6" />
  </svg>
);

export const ArrowIcon = ({ size = 18, color = "#5B6270" }: Props) => (
  <svg {...base(size, color)}>
    <path d="M5 12h14M13 6l6 6-6 6" />
  </svg>
);
