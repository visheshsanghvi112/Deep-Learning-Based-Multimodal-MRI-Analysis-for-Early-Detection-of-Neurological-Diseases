import { ImageResponse } from 'next/og'

// Image metadata
export const size = {
    width: 32,
    height: 32,
}
export const contentType = 'image/png'

// Image generation
export default function Icon() {
    return new ImageResponse(
        (
            <div
                style={{
                    fontSize: 24,
                    background: 'linear-gradient(135deg, #00d4ff 0%, #8b5cf6 50%, #ff00ff 100%)',
                    width: '100%',
                    height: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white',
                    borderRadius: '6px',
                    fontWeight: 'bold',
                    fontFamily: 'system-ui, sans-serif',
                }}
            >
                N
            </div>
        ),
        {
            ...size,
        }
    )
}
