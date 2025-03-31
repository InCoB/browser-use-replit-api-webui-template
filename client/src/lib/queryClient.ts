import { 
  QueryClient, 
  QueryFunction, 
  QueryKey, 
  QueryFilters,
  QueryCache,
  MutationCache
} from '@tanstack/react-query';
import { throwIfResNotOk } from '@/lib/utils';

/**
 * Custom API error class with enhanced information
 */
export class ApiError extends Error {
  status: number;
  data?: any;

  constructor(message: string, status: number, data?: any) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

/**
 * Configuration options for getQueryFn
 */
interface GetQueryFnOptions {
  /** How to handle 401 unauthorized responses */
  on401?: 'throw' | 'redirect';
  /** Base API URL to use for requests */
  baseUrl?: string;
  /** Additional headers to include with every request */
  headers?: Record<string, string>;
  /** Whether to parse JSON response (set to false for blob/text responses) */
  parseJson?: boolean;
}

/**
 * Creates a queryFn for react-query that handles API requests
 * 
 * @param options - Configuration options
 * @returns A queryFn compatible with react-query
 */
function getQueryFn({
  on401 = 'redirect',
  baseUrl = '/api',
  headers = {},
  parseJson = true
}: GetQueryFnOptions = {}): QueryFunction<any, QueryKey> {
  return async ({ queryKey, signal }) => {
    try {
      // The first element is the endpoint, the rest are params
      const [endpoint, ...params] = queryKey as string[];
      
      // Build query string from params
      const queryParams = params.length
        ? `?${params
            .filter(param => param != null)
            .map(param => param.toString().replace('=', '='))
            .join('&')}`
        : '';
      
      // Prepare request headers
      const requestHeaders = {
        'Content-Type': 'application/json',
        ...headers,
      };

      // Make API request
      const response = await fetch(`${baseUrl}/${endpoint}${queryParams}`, {
        headers: requestHeaders,
        credentials: 'include', // Include cookies in requests
        signal, // Pass the AbortSignal for query cancellation
      });
      
      // Handle 401 unauthorized based on configuration
      if (response.status === 401) {
        if (on401 === 'throw') {
          throw new ApiError('Unauthorized', 401);
        } else {
          // Redirect to login page
          window.location.href = '/login';
          return null;
        }
      }
      
      // Handle non-OK responses
      if (!response.ok) {
        let errorData;
        try {
          errorData = await response.json();
        } catch (e) {
          // If JSON parsing fails, use text
          errorData = await response.text();
        }

        throw new ApiError(
          `API request failed: ${response.statusText}`,
          response.status,
          errorData
        );
      }
      
      // Return appropriate response format
      if (!parseJson) {
        return response;
      }
      
      return response.json();
    } catch (error) {
      // Ensure all errors have a consistent format
      if (error instanceof ApiError) {
        throw error;
      }
      
      // Convert unknown errors to ApiError
      throw new ApiError(
        error instanceof Error ? error.message : 'Unknown error occurred',
        500
      );
    }
  };
}

/**
 * Default cache configuration - can be imported and customized
 */
export const defaultCacheConfig = {
  staleTime: 1000 * 60 * 5,     // 5 minutes
  gcTime: 1000 * 60 * 30,       // 30 minutes (previously cacheTime)
  retry: 3,
  retryDelay: (attemptIndex: number) => Math.min(1000 * 2 ** attemptIndex, 30000),
};

// Create QueryCache and MutationCache with proper error handlers
const queryCache = new QueryCache({
  onError: (error) => {
    console.error('[Query Error]', error);
  }
});

const mutationCache = new MutationCache({
  onError: (error) => {
    console.error('[Mutation Error]', error);
  }
});

/**
 * Creates the QueryClient instance with default configuration
 */
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      queryFn: getQueryFn({ on401: "throw" }),
      refetchInterval: false,
      refetchOnWindowFocus: false,
      staleTime: defaultCacheConfig.staleTime,
      gcTime: defaultCacheConfig.gcTime,
      retry: defaultCacheConfig.retry,
      retryDelay: defaultCacheConfig.retryDelay,
    },
    mutations: {
      retry: 1,
    }
  },
  queryCache,
  mutationCache
});

/**
 * Type-safe wrapper for invalidating queries
 */
export function invalidateQueries(queryKey: QueryKey, filters?: QueryFilters) {
  return queryClient.invalidateQueries({ queryKey, ...filters });
}

/**
 * Type-safe wrapper for prefetching queries
 */
export function prefetchQuery<T = unknown>(
  queryKey: QueryKey, 
  options?: any
) {
  return queryClient.prefetchQuery<T>({
    queryKey,
    ...options
  });
}

/**
 * Type-safe wrapper for setting query data
 */
export function setQueryData<T = unknown>(queryKey: QueryKey, data: T) {
  return queryClient.setQueryData<T>(queryKey, data);
}

/**
 * Type-safe wrapper for getting query data
 */
export function getQueryData<T = unknown>(queryKey: QueryKey) {
  return queryClient.getQueryData<T>(queryKey);
}

/**
 * Clears all queries from the cache
 */
export function clearQueries() {
  return queryClient.clear();
}

/**
 * Type-safe wrapper for fetching queries
 */
export function fetchQuery<T = unknown>(
  queryKey: QueryKey,
  options?: any
) {
  return queryClient.fetchQuery<T>({
    queryKey,
    ...options
  });
}

/**
 * Creates a mutation function for use with useMutation
 */
export function createMutation<TData = unknown, TVariables = unknown>(
  endpoint: string,
  method: 'POST' | 'PUT' | 'PATCH' | 'DELETE' = 'POST',
  options: {
    headers?: Record<string, string>;
    baseUrl?: string;
  } = {}
) {
  const { headers = {}, baseUrl = '/api' } = options;
  
  return async (variables: TVariables): Promise<TData> => {
    const response = await fetch(`${baseUrl}/${endpoint}`, {
      method,
      headers: {
        'Content-Type': 'application/json',
        ...headers,
      },
      credentials: 'include',
      body: JSON.stringify(variables),
    });
    
    if (!response.ok) {
      let errorData;
      try {
        errorData = await response.json();
      } catch (e) {
        errorData = await response.text();
      }
      
      throw new ApiError(
        `API ${method} request failed: ${response.statusText}`,
        response.status,
        errorData
      );
    }
    
    return response.json();
  };
}

// Enhanced query client with additional methods
const enhancedQueryClient = {
  ...queryClient,
  invalidateQueries,
  prefetchQuery,
  setQueryData,
  getQueryData,
  clearQueries,
  fetchQuery,
  createMutation,
};

export default enhancedQueryClient;