const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"

export async function fetchOasisSummary() {
  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 5000)
    
    const res = await fetch(`${API_BASE}/api/oasis1/summary`, {
      cache: "no-store",
      signal: controller.signal,
    })
    
    clearTimeout(timeoutId)
    if (!res.ok) throw new Error("Failed to fetch OASIS-1 summary")
    return res.json() as Promise<{
      dataset: string
      n_subjects: number
      n_cdr_labeled: number
      modality: string
    }>
  } catch (error) {
    // Return fallback data if API is not available
    console.warn("API not available, using fallback data:", error)
    return {
      dataset: "OASIS-1",
      n_subjects: 436,
      n_cdr_labeled: 280,
      modality: "T1-weighted structural MRI"
    }
  }
}

export async function fetchOasisSubjects(page = 1, pageSize = 3) {
  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 5000)
    
    const res = await fetch(
      `${API_BASE}/api/oasis1/subjects?page=${page}&page_size=${pageSize}`,
      { 
        cache: "no-store",
        signal: controller.signal,
      },
    )
    
    clearTimeout(timeoutId)
    if (!res.ok) throw new Error("Failed to fetch subjects")
    return res.json() as Promise<{
      items: Array<{
        subject_id: string
        age: number | null
        gender: string | null
        cdr: number | null
        mmse: number | null
        nwbv: number | null
        etiv: number | null
      }>
    }>
  } catch (error) {
    // Return fallback data if API is not available
    console.warn("API not available, using fallback data:", error)
    return {
      items: [
        {
          subject_id: "OAS1_0001_MR1",
          age: 76,
          gender: "M",
          cdr: 0.0,
          mmse: 29,
          nwbv: 0.736,
          etiv: 1983947
        },
        {
          subject_id: "OAS1_0002_MR1",
          age: 75,
          gender: "F",
          cdr: 0.5,
          mmse: 28,
          nwbv: 0.713,
          etiv: 2008292
        },
        {
          subject_id: "OAS1_0003_MR1",
          age: 72,
          gender: "M",
          cdr: 0.0,
          mmse: 30,
          nwbv: 0.701,
          etiv: 1973503
        }
      ]
    }
  }
}

export const API_CONFIG = {
  baseUrl: API_BASE,
}


