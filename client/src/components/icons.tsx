import { LucideProps } from "lucide-react";

export function YCombinatorLogo(props: LucideProps) {
  return (
    <svg
      width="38"
      height="38"
      viewBox="0 0 38 38"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <rect width="38" height="38" rx="19" fill="#FF6600" />
      <path
        d="M11.5 12H15.5L19.5 18.5V12H23.5V26H19.5L15.5 19.5V26H11.5V12Z"
        fill="white"
      />
    </svg>
  );
}

export function BrowserUseIcon(props: LucideProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <rect width="18" height="14" x="3" y="5" rx="2" />
      <path d="M21 8H3" />
      <circle cx="6" cy="11.5" r=".5" />
      <circle cx="9" cy="11.5" r=".5" />
      <path d="M12 16a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z" />
      <path d="m14 16 2 2" />
      <path d="M7 16h1" />
      <path d="M16 16h1" />
    </svg>
  );
}
